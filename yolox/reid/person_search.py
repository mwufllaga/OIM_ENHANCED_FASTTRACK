"""
Person Search Module - OIMNetPlus-based person detection and re-identification.

This module provides:
- PersonSearchModel: Main class for person detection and ReID feature extraction
- infer_batch: Batch inference function for pipeline integration
- extract_query_feature: Extract ReID feature from a given bounding box

Based on OIMNetPlus (ECCV 2022): https://github.com/cvlab-yonsei/OIMNetPlus

Usage:
    from ml_common.person_search import infer_batch, extract_query_feature
    
    # Gallery mode: detect all persons and extract features
    results = infer_batch([image], [{"det_conf": 0.5}])
    
    # Query mode: extract feature from a given bbox
    feature = extract_query_feature(image, {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.8})
"""
import os
import threading
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

try:
    from .util import MODEL_DIR, device
except ImportError:
    try:
        from util import MODEL_DIR, device
    except ImportError:
        # Standalone mode: inline device detection and model dir
        def _get_device():
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        device = _get_device()
        MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

logger = logging.getLogger(__name__)

# ============================================================================
# Hardcoded Configuration (from ssm.yaml + defaults.py)
# ============================================================================
DEFAULT_CONFIG = {
    # Input parameters
    "INPUT_MIN_SIZE": 900,
    "INPUT_MAX_SIZE": 1500,
    
    # Image normalization (ImageNet standard)
    "IMAGE_MEAN": [0.485, 0.456, 0.406],
    "IMAGE_STD": [0.229, 0.224, 0.225],
    
    # RPN parameters
    "RPN_NMS_THRESH": 0.7,
    "RPN_BATCH_SIZE_TRAIN": 256,
    "RPN_POS_FRAC_TRAIN": 0.5,
    "RPN_POS_THRESH_TRAIN": 0.7,
    "RPN_NEG_THRESH_TRAIN": 0.3,
    "RPN_PRE_NMS_TOPN_TRAIN": 12000,
    "RPN_PRE_NMS_TOPN_TEST": 6000,
    "RPN_POST_NMS_TOPN_TRAIN": 2000,
    "RPN_POST_NMS_TOPN_TEST": 300,
    
    # ROI Head parameters
    "ROI_BN_NECK": True,
    "ROI_NORM_TYPE": "protonorm",  # from ssm.yaml
    "ROI_AUGMENT": False,
    "ROI_BATCH_SIZE_TRAIN": 128,
    "ROI_POS_FRAC_TRAIN": 0.5,
    "ROI_POS_THRESH_TRAIN": 0.5,
    "ROI_NEG_THRESH_TRAIN": 0.5,
    "ROI_SCORE_THRESH_TEST": 0.5,
    "ROI_NMS_THRESH_TEST": 0.4,
    "ROI_DETECTIONS_PER_IMAGE_TEST": 300,
    
    # Loss parameters (needed for model structure initialization)
    "LOSS_TYPE": "LOIM",  # from ssm.yaml
    "LOSS_LUT_SIZE": 5532,
    "LOSS_CQ_SIZE": 5000,
    "LOSS_OIM_MOMENTUM": 0.5,
    "LOSS_OIM_SCALAR": 30.0,
    "LOSS_OIM_EPS": 0.1,  # from ssm.yaml
    
    # Output
    "EMBEDDING_DIM": 256,
}

# Model file path — use absolute path to ml_common weights
MODEL_PATH = os.getenv(
    "PERSON_SEARCH_MODEL_DIR",
    "/Users/liaoruoxing/Documents/program/mot/person_search_oim.pth"
)

# ============================================================================
# ProtoNorm Implementation (from custom_modules.py - inference only)
# ============================================================================
class _PrototypeNorm(_BatchNorm):
    """Prototype Normalization base class (inference mode only)."""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(_PrototypeNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.register_buffer("target_prototypes", None)
    
    def forward(self, input):
        # Inference mode: use running statistics
        assert self.target_prototypes is None, "Empty targets during testing with ProtoNorm"
        self._check_input_dim(input)
        ndim = input.ndim
        
        dim_o = [1] * ndim
        dim_o[1] = -1  # 1d: [1,-1] 2d: [1, -1, 1, 1]
        
        mean = self.running_mean
        var = self.running_var
        
        input = (input - mean.view(dim_o)) / (torch.sqrt(var.view(dim_o) + self.eps))
        if self.affine:
            input = input * self.weight.view(dim_o) + self.bias.view(dim_o)
        
        return input


class PrototypeNorm1d(_PrototypeNorm):
    """1D Prototype Normalization."""
    
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class PrototypeNorm2d(_PrototypeNorm):
    """2D Prototype Normalization."""
    
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


# ============================================================================
# Backbone Implementation (from resnet.py)
# ============================================================================
class Backbone(nn.Sequential):
    """ResNet backbone (conv1 -> layer3)."""
    
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict([
                ["conv1", resnet.conv1],
                ["bn1", resnet.bn1],
                ["relu", resnet.relu],
                ["maxpool", resnet.maxpool],
                ["layer1", resnet.layer1],  # res2
                ["layer2", resnet.layer2],  # res3
                ["layer3", resnet.layer3],  # res4
            ])
        )
        self.out_channels = 1024
    
    def forward(self, x):
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Sequential):
    """ResNet layer4 (res5) head."""
    
    def __init__(self, resnet):
        super(Res5Head, self).__init__(
            OrderedDict([["layer4", resnet.layer4]])
        )
        self.out_channels = [1024, 2048]
    
    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=False):
    """Build ResNet backbone and head."""
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    return Backbone(resnet), Res5Head(resnet)


# ============================================================================
# ReID Embedding Head (from base.py)
# ============================================================================
class ReIDEmbedding(nn.Module):
    """ReID embedding head that outputs 256-dim L2-normalized features."""
    
    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256, norm_type='none'):
        super(ReIDEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = int(dim)
        
        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            if norm_type == 'none':
                proj = nn.Sequential(
                    nn.Linear(in_channel, indv_dim, bias=False),
                )
                init.normal_(proj[0].weight, std=0.01)
            elif norm_type == 'protonorm':
                proj = nn.Sequential(
                    nn.Linear(in_channel, indv_dim, bias=False),
                    PrototypeNorm1d(indv_dim)
                )
                init.normal_(proj[0].weight, std=0.01)
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
            
            self.projectors[ftname] = proj
    
    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor]
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
        """
        outputs = []
        for k in self.featmap_names:
            v = featmaps[k]
            v = self._flatten_fc_input(v)
            outputs.append(self.projectors[k](v))
        return F.normalize(torch.cat(outputs, dim=1))
    
    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x
    
    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class BBoxPredictor(nn.Module):
    """Bounding box predictor for classification and regression."""
    
    def __init__(self, in_channels, num_classes, bn_neck=True):
        super(BBoxPredictor, self).__init__()
        
        # Classification
        self.bbox_cls = nn.Linear(in_channels, num_classes)
        init.normal_(self.bbox_cls.weight, std=0.01)
        init.constant_(self.bbox_cls.bias, 0)
        
        # Regression
        self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
        init.normal_(self.bbox_pred.weight, std=0.01)
        init.constant_(self.bbox_pred.bias, 0)
    
    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        
        bbox_scores = self.bbox_cls(x)
        bbox_deltas = self.bbox_pred(x)
        return bbox_scores, bbox_deltas


# ============================================================================
# ROI Heads (from base.py - inference only)
# ============================================================================
class PersonSearchRoIHeads(RoIHeads):
    """ROI heads for person search (inference only)."""
    
    def __init__(
        self,
        faster_rcnn_predictor,
        reid_head,
        norm_type,
        *args,
        **kwargs
    ):
        super(PersonSearchRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = ReIDEmbedding(
            featmap_names=['feat_res5'],
            in_channels=[2048],
            dim=256,
            norm_type=norm_type,
        )
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
    
    def forward(self, features, proposals, image_shapes, targets=None, query_img_as_gallery=False):
        """Inference forward pass."""
        roi_pooled_features = self.box_roi_pool(features, proposals, image_shapes)
        rcnn_features = self.reid_head(roi_pooled_features)
        class_logits, box_regression = self.box_predictor(rcnn_features['feat_res5'])
        embeddings_ = self.embedding_head(rcnn_features)
        
        result, losses = [], {}
        
        gt_det = None
        if query_img_as_gallery and targets is not None:
            gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings_}
        
        boxes, scores, embeddings, labels = self.postprocess_boxes(
            class_logits, box_regression, embeddings_,
            proposals, image_shapes, gt_det=gt_det
        )
        
        num_images = len(boxes)
        for i in range(num_images):
            result.append(dict(
                boxes=boxes[i],
                labels=labels[i],
                scores=scores[i],
                embeddings=embeddings[i],
            ))
        
        return result, losses
    
    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        gt_det=None,
    ):
        """Post-process detection boxes with NMS."""
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        
        if embeddings is not None:
            embeddings = embeddings.split(boxes_per_image, 0)
        
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        
        for n_iter, (boxes, scores, image_shape) in enumerate(zip(pred_boxes, pred_scores, image_shapes)):
            emb = embeddings[n_iter] if embeddings is not None else None
            
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            # Remove background predictions
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            
            # Remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            if emb is not None:
                emb = emb[inds]
            
            if gt_det is not None:
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                emb = torch.cat((emb, gt_det["embeddings"]), dim=0)
            
            # NMS
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if emb is not None:
                emb = emb[keep]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            if emb is not None:
                all_embeddings.append(emb)
        
        return all_boxes, all_scores, all_embeddings, all_labels


# ============================================================================
# Main Model (from base.py - inference only)
# ============================================================================
class PersonSearchNet(nn.Module):
    """OIMNetPlus person search network (inference only)."""
    
    def __init__(self, cfg=None):
        super(PersonSearchNet, self).__init__()
        
        if cfg is None:
            cfg = DEFAULT_CONFIG
        
        backbone, box_head = build_resnet(name="resnet50", pretrained=False)
        
        anchor_generator = AnchorGenerator(
            sizes=((8, 16, 32),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        
        pre_nms_top_n = dict(
            training=cfg["RPN_PRE_NMS_TOPN_TRAIN"],
            testing=cfg["RPN_PRE_NMS_TOPN_TEST"]
        )
        post_nms_top_n = dict(
            training=cfg["RPN_POST_NMS_TOPN_TRAIN"],
            testing=cfg["RPN_POST_NMS_TOPN_TEST"]
        )
        
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg["RPN_POS_THRESH_TRAIN"],
            bg_iou_thresh=cfg["RPN_NEG_THRESH_TRAIN"],
            batch_size_per_image=cfg["RPN_BATCH_SIZE_TRAIN"],
            positive_fraction=cfg["RPN_POS_FRAC_TRAIN"],
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg["RPN_NMS_THRESH"],
        )
        
        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        reid_head = deepcopy(box_head)
        
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = BBoxPredictor(2048, num_classes=2, bn_neck=cfg["ROI_BN_NECK"])
        
        roi_heads = PersonSearchRoIHeads(
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            norm_type=cfg["ROI_NORM_TYPE"],
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg["ROI_POS_THRESH_TRAIN"],
            bg_iou_thresh=cfg["ROI_NEG_THRESH_TRAIN"],
            batch_size_per_image=cfg["ROI_BATCH_SIZE_TRAIN"],
            positive_fraction=cfg["ROI_POS_FRAC_TRAIN"],
            bbox_reg_weights=None,
            score_thresh=cfg["ROI_SCORE_THRESH_TEST"],
            nms_thresh=cfg["ROI_NMS_THRESH_TEST"],
            detections_per_img=cfg["ROI_DETECTIONS_PER_IMAGE_TEST"],
        )
        
        transform = GeneralizedRCNNTransform(
            min_size=cfg["INPUT_MIN_SIZE"],
            max_size=cfg["INPUT_MAX_SIZE"],
            image_mean=cfg["IMAGE_MEAN"],
            image_std=cfg["IMAGE_STD"],
        )
        
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
    
    def forward(self, images, targets=None, query_img_as_gallery=False):
        """Forward pass (inference only)."""
        return self.inference(images, targets, query_img_as_gallery)
    
    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        Inference method.
        
        Args:
            images: List of tensors [C, H, W]
            targets: Optional list of dicts with "boxes" key for query mode
            query_img_as_gallery: If True, detect all people including GT box
        
        Returns:
            If targets is provided and not query_img_as_gallery:
                Tuple of embeddings for query boxes
            Else:
                List of detection dicts with boxes, scores, embeddings
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        
        if targets is not None and not query_img_as_gallery:
            # Query mode: extract features from given boxes
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features = self.roi_heads.reid_head(box_features)
            embeddings = self.roi_heads.embedding_head(box_features)
            return embeddings.split(1, 0)
        else:
            # Gallery mode: detect all persons
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(
                features, proposals, images.image_sizes, targets, query_img_as_gallery
            )
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections


# ============================================================================
# Model Wrapper with Lazy Loading
# ============================================================================
class PersonSearchModel:
    """
    Person search model wrapper with lazy loading.
    
    This class provides a simple interface for person detection and ReID.
    Models are only loaded on first use to reduce startup time.
    """
    
    def __init__(self):
        self._model = None
        self._device = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return
        
        logger.info("Loading PersonSearchNet model...")
        
        # Determine device
        self._device = torch.device(device)
        logger.info(f"Using device: {self._device}")
        
        # Create model
        self._model = PersonSearchNet(DEFAULT_CONFIG)
        
        # Load weights
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
        
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # Filter out reid_loss keys (not needed for inference)
        state_dict = {k: v for k, v in state_dict.items() if "reid_loss" not in k}
        
        self._model.load_state_dict(state_dict, strict=False)
        self._model.to(self._device)
        self._model.eval()
        
        self._loaded = True
        logger.info("PersonSearchNet model loaded successfully")
    
    def detect_and_extract(
        self,
        image: np.ndarray,
        det_conf: float = 0.5,
        max_det: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Detect persons and extract ReID features.
        
        Args:
            image: RGB numpy array [H, W, 3], uint8
            det_conf: Detection confidence threshold
            max_det: Maximum number of detections
        
        Returns:
            List of dicts with keys:
                - "person_embedding": List[float] (256-dim)
                - "boxes": {"x1", "y1", "x2", "y2"} (normalized to [0,1])
                - "confidence": float
        """
        self._ensure_loaded()
        
        h, w = image.shape[:2]
        
        # Convert numpy to tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self._device)
        
        with torch.no_grad():
            detections = self._model([img_tensor])
        
        results = []
        if len(detections) > 0:
            det = detections[0]
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            embeddings = det["embeddings"].cpu().numpy()
            
            # Filter by confidence and limit
            mask = scores >= det_conf
            boxes = boxes[mask]
            scores = scores[mask]
            embeddings = embeddings[mask]
            
            # Limit detections
            if len(boxes) > max_det:
                boxes = boxes[:max_det]
                scores = scores[:max_det]
                embeddings = embeddings[:max_det]
            
            for box, score, emb in zip(boxes, scores, embeddings):
                results.append({
                    "person_embedding": emb.tolist(),
                    "boxes": {
                        "x1": float(box[0]) / w,
                        "y1": float(box[1]) / h,
                        "x2": float(box[2]) / w,
                        "y2": float(box[3]) / h,
                    },
                    "confidence": float(score),
                })
        
        return results
    
    def extract_feature(
        self,
        image: np.ndarray,
        bbox: Dict[str, float]
    ) -> Optional[List[float]]:
        """
        Extract ReID feature from a given bounding box.
        
        Args:
            image: RGB numpy array [H, W, 3], uint8
            bbox: {"x1", "y1", "x2", "y2"} normalized coordinates [0,1]
        
        Returns:
            256-dim L2-normalized feature list, or None if failed
        """
        self._ensure_loaded()
        
        h, w = image.shape[:2]
        
        # Convert normalized bbox to pixel coordinates
        x1 = bbox["x1"] * w
        y1 = bbox["y1"] * h
        x2 = bbox["x2"] * w
        y2 = bbox["y2"] * h
        
        # Convert numpy to tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self._device)
        
        # Create target with boxes
        boxes_tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32).to(self._device)
        targets = [{"boxes": boxes_tensor}]
        
        with torch.no_grad():
            embeddings = self._model([img_tensor], targets)
        
        if embeddings and len(embeddings) > 0:
            emb = embeddings[0].cpu().numpy().flatten()
            return emb.tolist()
        
        return None


# ============================================================================
# Singleton Management
# ============================================================================
_PERSON_SEARCH_MODEL: Optional[PersonSearchModel] = None
_PERSON_SEARCH_LOCK = threading.Lock()


def _get_model() -> PersonSearchModel:
    """Get the singleton model instance."""
    global _PERSON_SEARCH_MODEL
    if _PERSON_SEARCH_MODEL is None:
        with _PERSON_SEARCH_LOCK:
            if _PERSON_SEARCH_MODEL is None:
                _PERSON_SEARCH_MODEL = PersonSearchModel()
    return _PERSON_SEARCH_MODEL


# ============================================================================
# Public API
# ============================================================================
def infer_batch(
    images: List[np.ndarray],
    configs: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """
    Batch inference for person detection and ReID feature extraction.
    
    Args:
        images: List of RGB numpy arrays [H, W, 3], uint8
        configs: List of config dicts for each image
            - "det_conf": float, detection confidence threshold (default: 0.5)
            - "max_det": int, maximum detections per image (default: 300)
    
    Returns:
        List of detection lists, one per image. Each detection is a dict:
            - "person_embedding": List[float] (256-dim L2-normalized)
            - "boxes": {"x1", "y1", "x2", "y2"} (normalized to [0,1])
            - "confidence": float
    
    Example:
        >>> results = infer_batch([image1, image2], [{"det_conf": 0.5}, {"det_conf": 0.6}])
        >>> print(f"Image 1: {len(results[0])} persons detected")
    """
    model = _get_model()
    
    all_results = []
    for image, config in zip(images, configs):
        det_conf = config.get("det_conf", 0.5)
        max_det = config.get("max_det", 300)
        results = model.detect_and_extract(image, det_conf=det_conf, max_det=max_det)
        all_results.append(results)
    
    return all_results


def extract_query_feature(
    image: np.ndarray,
    bbox: Dict[str, float]
) -> Optional[List[float]]:
    """
    Extract ReID feature from a given bounding box (query mode).
    
    Args:
        image: RGB numpy array [H, W, 3], uint8
        bbox: {"x1", "y1", "x2", "y2"} normalized coordinates [0,1]
    
    Returns:
        256-dim L2-normalized feature list, or None if extraction failed
    
    Example:
        >>> feature = extract_query_feature(image, {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.8})
        >>> if feature:
        ...     print(f"Feature dimension: {len(feature)}")
    """
    model = _get_model()
    return model.extract_feature(image, bbox)
