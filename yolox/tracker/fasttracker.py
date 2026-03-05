import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import json
import math
import cv2 as _cv2

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from yolox.reid import extract_query_feature as _reid_extract



class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False
        self.mean_history = []
        self.det_bbox_history = []  # (frame_id, tlbr) for ReID
        self.ref_embeddings = []  # list of ReID embeddings captured at frames 5, 10, 15
        self.ref_embedding_crops = []  # list of (frame_id, crop_img) for debug visualization
        self.ref_embedding_target_frames = {5, 10, 15}  # tracklet_len values to capture

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)

        self.det_bbox_history.append((frame_id, STrack.tlwh_to_tlbr(self._tlwh).copy()))
        if len(self.det_bbox_history) > 150:
            self.det_bbox_history.pop(0)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)

        self.det_bbox_history.append((frame_id, STrack.tlwh_to_tlbr(new_track.tlwh).copy()))
        if len(self.det_bbox_history) > 150:
            self.det_bbox_history.pop(0)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        
        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)

        self.det_bbox_history.append((frame_id, STrack.tlwh_to_tlbr(new_track.tlwh).copy()))
        if len(self.det_bbox_history) > 150:
            self.det_bbox_history.pop(0)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        
        # Mark for embedding capture (actual extraction happens in Fasttracker.update)
        if hasattr(self, 'ref_embedding_target_frames') and self.tracklet_len in self.ref_embedding_target_frames:
            self._pending_embedding_capture = True

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def is_occluded_by(box_a, box_b, iou_thresh=0.7):
    """Returns True if box_a is significantly overlapped by box_b"""
    inter = (
        max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) *
        max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    )
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    if area_a == 0:
        return False
    iou = inter / area_a
    return iou > iou_thresh

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

class Fasttracker(object):
    def __init__(self, args, config, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args

        self.det_thresh = config["track_thresh"]
        self.match_thresh = config["match_thresh"]
        self.buffer_size = int(frame_rate / 30.0 * config["track_buffer"])
        self.max_time_lost = self.buffer_size

        self.reset_velocity_offset_occ = config["reset_velocity_offset_occ"]
        self.reset_pos_offset_occ = config["reset_pos_offset_occ"]
        self.enlarge_bbox_occ = config["enlarge_bbox_occ"]
        self.dampen_motion_occ = config["dampen_motion_occ"]
        self.active_occ_to_lost_thresh = config["active_occ_to_lost_thresh"]
        self.init_iou_suppress = config["init_iou_suppress"]

        # --- Dynamic ROI loading ---
        self.roi_points = []
        self.theta_values = []
        rois = config.get("ROIs", {})

        for name, pts in rois.items():
            try:
                roi_np = np.array(pts)
                self.roi_points.append(roi_np)
                theta = self.compute_theta(roi_np)
                self.theta_values.append(theta)
                print(f"[ROI] {name} loaded with theta = {theta:.2f} degrees.")
            except Exception as e:
                print(f"[Warning] Failed to load {name}: {e}")

        self.roi_repair_max_gap = config.get("roi_repair_max_gap", 15)
        self.dir_window_N = config.get("dir_window_N", 10)
        self.dir_margin_deg = config.get("dir_margin_deg", 2.0)

        self.kalman_filter = KalmanFilter()

        # ReID split handling (person_search is a hard dependency)
        self.reid_enabled = config.get("reid_enabled", True)
        self.reid_sim_thresh = config.get("reid_sim_thresh", 0.4)
        self.reid_split_iou_thresh = config.get("reid_split_iou_thresh", 0.3)  # IoU > this = overlapping
        self.reid_separation_iou_thresh = config.get("reid_separation_iou_thresh", 0.1)  # IoU < this = separated, trigger ReID
        self.frame_buffer = deque(maxlen=30)   # (frame_id, raw_img_rgb)
        self.frame_sample_interval = 5
        self.reid_vis_boxes = []  # Visual feedback: list of (tlbr, track_id, label_str) for current frame
        # overlapped_pairs: set of (tid_a, tid_b) where tid_a < tid_b, tracks that have overlapped
        self.overlapped_pairs = set()
        # Debug: save ReID crops
        self.reid_debug_dir = config.get("reid_debug_dir", None)
        if self.reid_debug_dir:
            os.makedirs(self.reid_debug_dir, exist_ok=True)
            print(f"[ReID] Debug crops will be saved to: {self.reid_debug_dir}")
        if self.reid_enabled:
            print("[ReID] Split detection ENABLED")

        # Print config in terminal
        print("=== FastTracker Config ===")
        print(json.dumps(config, indent=2))
        print("=============================")

    def update(self, output_results, img_info, img_size, raw_img=None):
        self.frame_id += 1

        # Store every frame for ReID (buffer maxlen=30 covers the ~18-frame maintaining window)
        if raw_img is not None and self.reid_enabled:
            self.frame_buffer.append((self.frame_id, _cv2.cvtColor(raw_img, _cv2.COLOR_BGR2RGB)))

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.det_thresh
        inds_low = scores > 0.25
        inds_high = scores < self.det_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)

        # ContainPenalty: when a track with not_matched>0 has an inflated KF box
        # that partially contains a detection, IoU can be misleadingly low.
        # Use containment ratio as a proxy cost to allow the match.
        if len(strack_pool) > 0 and len(detections) > 0:
            for i in range(len(strack_pool)):
                if strack_pool[i].not_matched == 0:
                    continue
                t_box = strack_pool[i].tlbr
                for j in range(len(detections)):
                    d_box = STrack.tlwh_to_tlbr(detections[j].tlwh)
                    d_area = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
                    ix1 = max(t_box[0], d_box[0])
                    iy1 = max(t_box[1], d_box[1])
                    ix2 = min(t_box[2], d_box[2])
                    iy2 = min(t_box[3], d_box[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    contain_ratio = inter / (d_area + 1e-9)
                    if contain_ratio > 0.3:
                        new_dist = 1.0 - contain_ratio
                        if new_dist < dists[i][j]:
                            print(f"[ContainPenalty] Frame {self.frame_id}: "
                                  f"T{strack_pool[i].track_id} "
                                  f"(not_matched={strack_pool[i].not_matched}) "
                                  f"contains det (ratio={contain_ratio:.2f}), "
                                  f"dist {dists[i][j]:.3f} -> {new_dist:.3f}")
                            dists[i][j] = new_dist

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        # ReID visual feedback reset
        self.reid_vis_boxes = []

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            track.is_occluded = False
            track.not_matched =0
            track.occluded_len =0

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

            ## The tracklet is rematched with one DET, so it is not occluded    
            track.is_occluded = False
            track.not_matched =0
            track.occluded_len =0

        ## occlusion handling version
        for it in u_track:
            track = r_tracked_stracks[it]
            track.not_matched += 1

            # Try detecting occlusion
            if not track.is_occluded and track.state == TrackState.Tracked:
                for other in activated_starcks:
                    if track.track_id == other.track_id:
                        continue
                    if not other.is_activated or other.is_occluded:
                        continue

                    if is_occluded_by(track.tlbr, other.tlbr):
                        track.is_occluded = True
                        track.occluded_len += 1
                        track.last_occluded_frame = self.frame_id
                        track.was_recently_occluded = True

                        # Reset velocity
                        if len(track.mean_history) >= self.reset_velocity_offset_occ:
                            old_mean = track.mean_history[-self.reset_velocity_offset_occ]
                            track.mean[4:8] = old_mean[4:8]

                        # Reset position
                        if len(track.mean_history) >= self.reset_pos_offset_occ:
                            old_mean = track.mean_history[-self.reset_pos_offset_occ]
                            track.mean[0:4] = old_mean[0:4]

                        # Enlarge once
                        if track.occluded_len == 1:
                            track.mean[3] *= self.enlarge_bbox_occ  # increase height
                            # track.mean[2] = track.mean[2] / track.mean[3]  # adjust aspect ratio

                        # Dampen motion
                        track.mean[4:8] *= self.dampen_motion_occ
                        break

            # Handle occlusion flags
            if not track.is_occluded:
                track.occluded_len = 0
            else:
                track.occluded_len += 1

            if track.was_recently_occluded and (self.frame_id - track.last_occluded_frame > 40):
                track.was_recently_occluded = False

            # Finally decide whether to mark as lost
            if track.state != TrackState.Lost:
                if track.not_matched > 2 and (
                    not track.is_occluded or track.occluded_len > self.active_occ_to_lost_thresh
                ):
                    track.mark_lost()
                    lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_lost()
            lost_stracks.append(track)
            
        # After handling Stage-2 matches and updating tracks enforece environment constraints:
        for t in activated_starcks:
            self.enforce_environment_constraints(t)
        for t in refind_stracks:
            self.enforce_environment_constraints(t)
        for t in self.tracked_stracks:
            if t.state == TrackState.Tracked and t not in activated_starcks and t not in refind_stracks:
                self.enforce_environment_constraints(t)

        """ Step 4: Init new stracks (with IoU suppression) """
        # Gather active tracks *now* (already-updated ones + still-tracked ones)
        active_now = {t.track_id: t for t in self.tracked_stracks if t.state == TrackState.Tracked}
        for t in activated_starcks:
            active_now[t.track_id] = t
        active_now = list(active_now.values())

        init_iou_thr = getattr(self, "init_iou_suppress", None)
        if init_iou_thr is None:
            print("Warn, init not found")
            init_iou_thr = 0.8

        # --- ReID recovery: collect all (det, candidate_track, sim) first ---
        # Then for each candidate track, pick the detection with highest sim.
        # This prevents the same track being recovered by multiple detections.
        reid_proposals = []  # list of (det_index, det_STrack, candidate_track, sim)
        non_overlap_dets = []  # dets that pass IoU suppress but need ReID or activate

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue

            # compute max IoU with any active track this frame
            det_box = STrack.tlwh_to_tlbr(track.tlwh)
            max_iou = 0.0
            for at in active_now:
                at_box = at.tlbr  # already tlbr
                max_iou = max(max_iou, _iou(det_box, at_box))
                if max_iou >= init_iou_thr:
                    break

            # Only consider if it does NOT heavily overlap an active track
            if max_iou < init_iou_thr:
                non_overlap_dets.append(track)

        # Phase 1: Collect ReID match proposals for all non-overlap dets
        if self.reid_enabled and raw_img is not None and non_overlap_dets:
            all_proposals = self._batch_recover_maintaining_tracks(
                non_overlap_dets, raw_img, r_tracked_stracks, lost_stracks
            )
            # all_proposals: list of (det_STrack, candidate_track, sim)
            # For each candidate track, keep only the det with highest sim
            best_per_track = {}  # track_id -> (det_STrack, candidate_track, sim)
            for det_st, cand_t, sim in all_proposals:
                tid = cand_t.track_id
                if tid not in best_per_track or sim > best_per_track[tid][2]:
                    best_per_track[tid] = (det_st, cand_t, sim)
            # Also ensure each det is used at most once (pick highest sim)
            used_dets = set()
            for tid, (det_st, cand_t, sim) in sorted(best_per_track.items(), key=lambda x: -x[1][2]):
                det_id_key = id(det_st)
                if det_id_key in used_dets:
                    continue
                used_dets.add(det_id_key)
                reid_proposals.append((det_st, cand_t, sim))

        # Phase 2: Apply recoveries and activate remaining dets
        recovered_det_ids = {id(det_st) for det_st, _, _ in reid_proposals}
        for det_st, cand_t, sim in reid_proposals:
            det_tlbr = STrack.tlwh_to_tlbr(det_st.tlwh)
            self.reid_vis_boxes.append((
                det_tlbr.copy(), cand_t.track_id,
                f"Recover T{cand_t.track_id} sim={sim:.2f}"
            ))
            cand_t.re_activate(det_st, self.frame_id, new_id=False)
            cand_t.not_matched = 0
            cand_t.is_occluded = False
            cand_t.occluded_len = 0
            refind_stracks.append(cand_t)

        for track in non_overlap_dets:
            if id(track) not in recovered_det_ids:
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            recently_occluded = (
                track.was_recently_occluded and
                (self.frame_id - track.last_occluded_frame <= 40)  # configurable if needed
            )

            if not recently_occluded and (self.frame_id - track.end_frame > self.max_time_lost):
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # ReID overlap tracking, separation check, then ref embedding capture
        if self.reid_enabled and raw_img is not None:
            self._record_overlaps(output_stracks)
            self._check_separations(raw_img, output_stracks)
            self._capture_ref_embeddings(raw_img, output_stracks)

        return output_stracks

    def _find_containing_boxes(self, detections, contain_thresh=0.70, area_ratio_thresh=1.3):
        """Find detection indices that are 'containing boxes' (oversized).

        A detection is flagged if:
        1. It contains > contain_thresh of a smaller detection's area
        2. Its area is > area_ratio_thresh times the smaller detection's area

        Returns a set of indices into detections that should receive a
        distance penalty (but are NOT removed).
        """
        n = len(detections)
        if n < 2:
            return set()

        tlbrs = [STrack.tlwh_to_tlbr(d.tlwh) for d in detections]
        areas = [d.tlwh[2] * d.tlwh[3] for d in detections]

        flagged = set()
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                # Only check if i is the LARGER box
                if areas[i] <= areas[j]:
                    continue

                # Compute intersection
                ix1 = max(tlbrs[i][0], tlbrs[j][0])
                iy1 = max(tlbrs[i][1], tlbrs[j][1])
                ix2 = min(tlbrs[i][2], tlbrs[j][2])
                iy2 = min(tlbrs[i][3], tlbrs[j][3])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih

                if inter == 0:
                    continue

                contain_ratio = inter / (areas[j] + 1e-9)
                area_ratio = areas[i] / (areas[j] + 1e-9)

                if contain_ratio > contain_thresh and area_ratio > area_ratio_thresh:
                    flagged.add(i)
                    print(f"[ContainPenalty] Frame {self.frame_id}: det[{i}] "
                          f"(area={areas[i]:.0f}) contains det[{j}] (area={areas[j]:.0f}), "
                          f"contain={contain_ratio:.2f}, ratio={area_ratio:.2f}")
                    break  # already flagged i

        return flagged

    # ================================================================
    # Maintaining-track recovery via ReID
    # ================================================================

    def _batch_recover_maintaining_tracks(self, new_dets, raw_img, unmatched_tracked, just_lost,
                                           iou_thresh=0.3, sim_thresh=0.6):
        """Batch version: for all new detections, find ReID recovery proposals.

        Returns list of (det_STrack, candidate_track, sim) where sim > sim_thresh.
        Caller is responsible for resolving conflicts (multiple dets -> same track).
        """
        # Collect maintaining candidates (shared across all dets)
        candidates = []
        for t in unmatched_tracked:
            if t.not_matched > 0 and t.state == TrackState.Tracked:
                candidates.append(t)
        for t in just_lost:
            candidates.append(t)
        for t in self.lost_stracks:
            if (self.frame_id - t.end_frame) <= self.max_time_lost:
                if t not in just_lost:
                    candidates.append(t)

        if not candidates:
            return []

        # Pre-compute candidate embeddings (extract once, reuse for all dets)
        stored_frames = {fid: img for fid, img in self.frame_buffer}
        cand_embeddings = {}  # track_id -> embedding
        for t in candidates:
            if t.track_id not in cand_embeddings:
                emb = self._extract_last_det_embedding(t, stored_frames)
                if emb is not None:
                    cand_embeddings[t.track_id] = (t, emb)

        if not cand_embeddings:
            return []

        rgb_img_cur = _cv2.cvtColor(raw_img, _cv2.COLOR_BGR2RGB)
        proposals = []

        for det in new_dets:
            det_tlbr = STrack.tlwh_to_tlbr(det.tlwh)
            det_cx = (det_tlbr[0] + det_tlbr[2]) / 2
            det_cy = (det_tlbr[1] + det_tlbr[3]) / 2
            det_w = det_tlbr[2] - det_tlbr[0]
            det_h = det_tlbr[3] - det_tlbr[1]
            dist_thresh = max(det_w, det_h) * 1.5

            # Find spatially nearby candidates
            nearby = []
            for tid, (t, t_emb) in cand_embeddings.items():
                t_tlbr = t.tlbr
                iou_val = _iou(det_tlbr, t_tlbr)

                ix1 = max(det_tlbr[0], t_tlbr[0])
                iy1 = max(det_tlbr[1], t_tlbr[1])
                ix2 = min(det_tlbr[2], t_tlbr[2])
                iy2 = min(det_tlbr[3], t_tlbr[3])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                det_area = (det_tlbr[2] - det_tlbr[0]) * (det_tlbr[3] - det_tlbr[1])
                contain_ratio = inter / (det_area + 1e-9) if det_area > 0 else 0.0

                t_cx = (t_tlbr[0] + t_tlbr[2]) / 2
                t_cy = (t_tlbr[1] + t_tlbr[3]) / 2
                center_dist = ((det_cx - t_cx) ** 2 + (det_cy - t_cy) ** 2) ** 0.5

                if iou_val > iou_thresh or contain_ratio > 0.3 or center_dist < dist_thresh:
                    nearby.append((t, t_emb))

            if not nearby:
                continue

            new_emb = self._extract_embedding(rgb_img_cur, det_tlbr)
            if new_emb is None:
                continue

            for t, t_emb in nearby:
                sim = float(np.dot(new_emb, t_emb))
                if sim > sim_thresh:
                    print(f"[ReID-Recover] Frame {self.frame_id}: det matched "
                          f"track {t.track_id} (sim={sim:.3f}, "
                          f"not_matched={t.not_matched}, "
                          f"lost_frames={self.frame_id - t.end_frame})")
                    proposals.append((det, t, sim))

        return proposals

    def _extract_last_det_embedding(self, track, stored_frames):
        """Extract embedding from a track's last successful detection.
        
        Searches det_bbox_history in reverse (newest first) to find a frame
        that exists in stored_frames (frame_buffer). Extracts embedding using
        that frame image and the detection bbox at that time.
        
        Returns 256-dim embedding or None.
        """
        if not hasattr(track, 'det_bbox_history') or not track.det_bbox_history:
            return None
        # Reverse: prefer the most recent detection
        for fid, tlbr in reversed(track.det_bbox_history):
            if fid in stored_frames:
                rgb_img = stored_frames[fid]
                emb = self._extract_embedding(rgb_img, tlbr)
                if emb is not None:
                    return emb
        return None

    # ================================================================
    # ReID-based split detection helpers
    # ================================================================

    REID_CROP_SIZE = 400  # crop a 400x400 region centered on bbox before ReID

    def _extract_embedding(self, rgb_img, tlbr):
        """Extract 256-dim L2-normalized ReID embedding from an RGB image.
        
        Crops a REID_CROP_SIZE x REID_CROP_SIZE region centered on the bbox
        before passing to person_search, so the bbox occupies a reasonable
        fraction of the input and background doesn't dominate the feature.
        """
        h, w = rgb_img.shape[:2]
        x1, y1, x2, y2 = float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])
        bw, bh = x2 - x1, y2 - y1
        if bw < 1 or bh < 1:
            return None

        # Center of bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half = self.REID_CROP_SIZE / 2

        # Crop bounds (clamp to image)
        crop_x1 = int(max(0, cx - half))
        crop_y1 = int(max(0, cy - half))
        crop_x2 = int(min(w, cx + half))
        crop_y2 = int(min(h, cy + half))

        crop = rgb_img[crop_y1:crop_y2, crop_x1:crop_x2]
        ch, cw = crop.shape[:2]
        if ch < 10 or cw < 10:
            return None

        # Translate bbox to crop coordinates and normalize
        bbox_norm = {
            "x1": max(0.0, (x1 - crop_x1) / cw),
            "y1": max(0.0, (y1 - crop_y1) / ch),
            "x2": min(1.0, (x2 - crop_x1) / cw),
            "y2": min(1.0, (y2 - crop_y1) / ch),
        }
        if bbox_norm["x2"] - bbox_norm["x1"] < 0.01 or bbox_norm["y2"] - bbox_norm["y1"] < 0.01:
            return None
        try:
            feat = _reid_extract(crop, bbox_norm)
            if feat is not None:
                return np.array(feat, dtype=np.float32)
        except Exception as e:
            print(f"[ReID] Embedding extraction failed: {e}")
        return None

    def _crop_image(self, img, tlbr):
        """Crop a region from image given tlbr bbox. Returns BGR crop or None."""
        h, w = img.shape[:2]
        x1 = max(0, int(tlbr[0]))
        y1 = max(0, int(tlbr[1]))
        x2 = min(w, int(tlbr[2]))
        y2 = min(h, int(tlbr[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2].copy()

    def _find_track_embedding_data(self, track):
        """Find a stored frame + bbox for this track's historical appearance.
        Prefers the OLDEST available stored frame (most reliable identity,
        least likely contaminated by an ID switch).
        Returns (rgb_img, tlbr) or (None, None)."""
        if not hasattr(track, 'det_bbox_history') or not track.det_bbox_history:
            return None, None
        stored = {fid: img for fid, img in self.frame_buffer}
        if not stored:
            return None, None
        # Iterate from OLDEST to newest → prefer oldest frame (most reliable identity)
        for fid, tlbr in track.det_bbox_history:
            if fid in stored:
                return stored[fid], tlbr
        return None, None

    def _record_overlaps(self, output_stracks):
        """Record track pairs that have IoU > threshold (overlapping).
        
        These pairs need to be checked when they separate.
        """
        if len(output_stracks) < 2:
            return
        
        # Build list of (track_id, tlbr) — include all tracks with tracklet_len >= 2
        track_boxes = []
        for t in output_stracks:
            if t.tracklet_len < 2:
                continue  # Skip very new tracks (just initialized)
            track_boxes.append((t.track_id, t.tlbr, t))
        
        # Check all pairs
        for i in range(len(track_boxes)):
            for j in range(i + 1, len(track_boxes)):
                tid_a, tlbr_a, _ = track_boxes[i]
                tid_b, tlbr_b, _ = track_boxes[j]
                iou = _iou(tlbr_a, tlbr_b)
                
                if iou > self.reid_split_iou_thresh:
                    # Record this pair (use sorted tuple for consistency)
                    pair = (min(tid_a, tid_b), max(tid_a, tid_b))
                    if pair not in self.overlapped_pairs:
                        self.overlapped_pairs.add(pair)
                        print(f"[ReID] Frame {self.frame_id}: T{tid_a} and T{tid_b} overlapping (IoU={iou:.2f})")

    def _check_separations(self, raw_img, output_stracks):
        """Check if any overlapped pairs have separated.
        
        When a pair separates (IoU <= threshold), use ReID to verify IDs.
        Unified voting logic: each ref set votes independently.
        - 0 ref sets = no vote, no swap
        - 1 ref set = follow that vote
        - 2 ref sets = if agree, follow; if conflict, follow larger margin
        """
        if not self.overlapped_pairs:
            return
        
        raw_rgb = _cv2.cvtColor(raw_img, _cv2.COLOR_BGR2RGB)
        
        # Build track lookup
        track_by_id = {t.track_id: t for t in output_stracks}
        
        resolved_pairs = []
        
        for pair in list(self.overlapped_pairs):
            tid_a, tid_b = pair
            
            # Check if both tracks still exist
            if tid_a not in track_by_id or tid_b not in track_by_id:
                resolved_pairs.append(pair)
                continue
            
            track_a = track_by_id[tid_a]
            track_b = track_by_id[tid_b]
            
            # Calculate current IoU
            iou = _iou(track_a.tlbr, track_b.tlbr)
            
            if iou >= self.reid_separation_iou_thresh:
                # Not fully separated yet, keep waiting
                continue
            
            # Separated!
            print(f"[ReID] Frame {self.frame_id}: T{tid_a} and T{tid_b} separated (IoU={iou:.2f})")
            
            # Extract current embeddings for both positions
            emb_a = self._extract_embedding(raw_rgb, track_a.tlbr)
            emb_b = self._extract_embedding(raw_rgb, track_b.tlbr)
            
            # Save debug crops
            if self.reid_debug_dir:
                crop_a = self._crop_image(raw_img, track_a.tlbr)
                crop_b = self._crop_image(raw_img, track_b.tlbr)
                if crop_a is not None:
                    _cv2.imwrite(os.path.join(self.reid_debug_dir, f"cmp_f{self.frame_id}_T{tid_a}_current.jpg"), crop_a)
                if crop_b is not None:
                    _cv2.imwrite(os.path.join(self.reid_debug_dir, f"cmp_f{self.frame_id}_T{tid_b}_current.jpg"), crop_b)
                if hasattr(track_a, 'ref_embedding_crops'):
                    for i, (fid, crop) in enumerate(track_a.ref_embedding_crops):
                        _cv2.imwrite(os.path.join(self.reid_debug_dir, f"cmp_f{self.frame_id}_T{tid_a}_ref{i+1}_from_f{fid}.jpg"), crop)
                if hasattr(track_b, 'ref_embedding_crops'):
                    for i, (fid, crop) in enumerate(track_b.ref_embedding_crops):
                        _cv2.imwrite(os.path.join(self.reid_debug_dir, f"cmp_f{self.frame_id}_T{tid_b}_ref{i+1}_from_f{fid}.jpg"), crop)
            
            if emb_a is None or emb_b is None:
                print(f"[ReID] Cannot extract current embeddings, skipping")
                resolved_pairs.append(pair)
                continue
            
            # Gather ref_embeddings (may be empty for one or both)
            refs_a = track_a.ref_embeddings if hasattr(track_a, 'ref_embeddings') else []
            refs_b = track_b.ref_embeddings if hasattr(track_b, 'ref_embeddings') else []
            
            # Unified voting: each ref set that exists casts an independent vote
            # vote: +1 = IDs are correct, -1 = IDs are swapped
            votes = []
            
            if len(refs_a) > 0:
                # refs_a belongs to whoever should be at position A
                sim_pos_a = max(float(np.dot(emb_a, ref)) for ref in refs_a)
                sim_pos_b = max(float(np.dot(emb_b, ref)) for ref in refs_a)
                margin = abs(sim_pos_a - sim_pos_b)
                vote = +1 if sim_pos_a >= sim_pos_b else -1
                votes.append((vote, margin))
                print(f"[ReID] refs_A vote: pos_A_sim={sim_pos_a:.3f}, pos_B_sim={sim_pos_b:.3f} -> {'correct' if vote > 0 else 'swapped'} (margin={margin:.3f})")
            
            if len(refs_b) > 0:
                # refs_b belongs to whoever should be at position B
                sim_pos_a = max(float(np.dot(emb_a, ref)) for ref in refs_b)
                sim_pos_b = max(float(np.dot(emb_b, ref)) for ref in refs_b)
                margin = abs(sim_pos_b - sim_pos_a)
                vote = +1 if sim_pos_b >= sim_pos_a else -1
                votes.append((vote, margin))
                print(f"[ReID] refs_B vote: pos_A_sim={sim_pos_a:.3f}, pos_B_sim={sim_pos_b:.3f} -> {'correct' if vote > 0 else 'swapped'} (margin={margin:.3f})")
            
            if len(votes) == 0:
                print(f"[ReID] No ref_embeddings for either track, cannot decide")
                resolved_pairs.append(pair)
                continue
            
            # Determine final vote
            if len(votes) == 1:
                final_vote = votes[0][0]
            else:
                # 2 votes
                if votes[0][0] == votes[1][0]:
                    # Both agree
                    final_vote = votes[0][0]
                else:
                    # Conflict: follow the vote with larger margin
                    final_vote = votes[0][0] if votes[0][1] >= votes[1][1] else votes[1][0]
                    print(f"[ReID] Vote conflict! margins={votes[0][1]:.3f} vs {votes[1][1]:.3f}, following larger")
            
            # Visual feedback
            self.reid_vis_boxes.append((track_a.tlbr.copy(), tid_a,
                                        f"T{tid_a} ReID: {'SWAP' if final_vote < 0 else 'OK'}"))
            self.reid_vis_boxes.append((track_b.tlbr.copy(), tid_b,
                                        f"T{tid_b} ReID: {'SWAP' if final_vote < 0 else 'OK'}"))
            
            if final_vote < 0:
                # Swap needed
                print(f"[ReID] SWAP DETECTED! T{tid_a} <-> T{tid_b}")
                track_a.track_id, track_b.track_id = track_b.track_id, track_a.track_id
                track_a.ref_embeddings, track_b.ref_embeddings = track_b.ref_embeddings, track_a.ref_embeddings
                if hasattr(track_a, 'ref_embedding_crops') and hasattr(track_b, 'ref_embedding_crops'):
                    track_a.ref_embedding_crops, track_b.ref_embedding_crops = track_b.ref_embedding_crops, track_a.ref_embedding_crops
                print(f"[ReID] Swapped: now T{track_a.track_id} and T{track_b.track_id}")
            else:
                print(f"[ReID] IDs are correct")
            
            resolved_pairs.append(pair)
        
        # Remove resolved pairs
        for pair in resolved_pairs:
            self.overlapped_pairs.discard(pair)

    def _capture_ref_embeddings(self, raw_img, output_stracks):
        """Capture ref embeddings for tracks with pending captures.
        
        Called AFTER _record_overlaps and _check_separations.
        Skips tracks currently in overlapped_pairs (refs captured during
        overlap are unreliable). Keeps _pending_embedding_capture=True
        so capture happens after separation.
        """
        # Build set of track_ids currently in any overlapped pair
        overlapped_tids = set()
        for tid_a, tid_b in self.overlapped_pairs:
            overlapped_tids.add(tid_a)
            overlapped_tids.add(tid_b)
        
        raw_rgb = _cv2.cvtColor(raw_img, _cv2.COLOR_BGR2RGB)
        
        for track in output_stracks:
            if not (hasattr(track, '_pending_embedding_capture') and track._pending_embedding_capture):
                continue
            if len(track.ref_embeddings) >= 3:
                track._pending_embedding_capture = False
                continue
            
            # Skip if track is in any overlapped pair — keep pending for later
            if track.track_id in overlapped_tids:
                continue
            
            # Safe to capture
            track._pending_embedding_capture = False
            tlbr = track.tlbr
            emb = self._extract_embedding(raw_rgb, tlbr)
            if emb is not None:
                track.ref_embeddings.append(emb)
                crop = self._crop_image(raw_img, tlbr)
                if crop is not None:
                    track.ref_embedding_crops.append((self.frame_id, crop))
                    if self.reid_debug_dir:
                        crop_path = os.path.join(self.reid_debug_dir, f"ref_T{track.track_id}_emb{len(track.ref_embeddings)}_f{self.frame_id}.jpg")
                        _cv2.imwrite(crop_path, crop)
                print(f"[ReID] T{track.track_id}: captured ref_embedding {len(track.ref_embeddings)}/3 at tracklet_len={track.tracklet_len}")

    def enforce_environment_constraints(self, t):
        """
        Enforce ROI containment and cone direction for a single track t.
        Assumes t.tlwh exists and t.history (list of tlwh or centers) exists/updated per frame.
        """
        if not self.roi_points:
            return

        # Ensure we have a minimal trajectory history (store centers per frame)
        if not hasattr(t, "center_history"):
            t.center_history = []
        curr_center = self._get_center_from_tlwh(t.tlwh)
        t.center_history.append(curr_center.copy())

        # 1) Determine which ROI contains the current center (if any)
        roi_idx = -1
        for i, roi in enumerate(self.roi_points):
            if len(roi) >= 3 and self._point_in_polygon(curr_center, roi):
                roi_idx = i
                break
        if roi_idx < 0:
            # Not inside any ROI -> no constraint
            return

        roi = self.roi_points[roi_idx]

        # ==========================================================
        # ROI History Repair: if the track is currently inside ROI,
        # but had a short out-of-ROI excursion in the past few frames,
        # project those out-of-bound points back onto the ROI boundary.
        # ==========================================================

        if self._point_in_polygon(curr_center, roi):
            # Track is inside ROI -> check its recent history
            if len(t.center_history) > 2:
                last_inside_idx = None
                last_outside_idx = None

                # Iterate backward through history to find last in/out transitions
                for i in range(len(t.center_history) - 2, -1, -1):
                    pt = t.center_history[i]
                    inside = self._point_in_polygon(pt, roi)

                    if inside and last_outside_idx is not None:
                        # Found transition from outside -> inside
                        last_inside_idx = i
                        break
                    if not inside and last_outside_idx is None:
                        # First time we see an outside segment
                        last_outside_idx = i

                # If we found an out-of-ROI segment that ended recently
                if last_outside_idx is not None and last_inside_idx is not None:
                    gap = last_outside_idx - last_inside_idx
                    if 0 < gap <= self.roi_repair_max_gap:
                        # Repair the short outside segment
                        for j in range(last_inside_idx + 1, last_outside_idx + 1):
                            pt_out = t.center_history[j]
                            # Clamp each point back to the ROI boundary
                            clamped_point = self._clamp_point_to_polygon(pt_out, roi)

                            # Update both geometric and KF mean history
                            t.center_history[j] = clamped_point
                            if hasattr(t, "mean_history") and j < len(t.mean_history):
                                t.mean_history[j][:2] = clamped_point

                        # Update the track’s current center (last frame)
                        curr_center = t.center_history[-1]
                        x, y, w, h = t.tlwh
                        new_x = curr_center[0] - 0.5 * w
                        new_y = curr_center[1] - 0.5 * h
                        t.mean[0:2] = np.array([new_x, new_y], dtype=float)

                        print(f"[ROI-Repair] Track {t.track_id}: repaired short excursion ({gap} frames).")

        # 2) Direction cone enforcement
        if len(roi) == 4:
            axis_u, theta_deg = self._cone_axis_and_theta(roi)
        else:
            # If not a quad, skip direction constraint
            return

        N = self.dir_window_N
        if len(t.center_history) >= (N + 1):
            pk   = t.center_history[-1]
            pk_N = t.center_history[-1 - N]
            delta = pk - pk_N
            if np.linalg.norm(delta) > 1e-6:
                # Compare phi vs theta/2 -> if violated, rotate last step to boundary
                prev = t.center_history[-2]
                adjusted = self._clamp_to_cone(pk_N, pk, axis_u, theta_deg)
                if not np.allclose(adjusted, pk, atol=1e-3):
                    # Apply adjusted position
                    t.center_history[-1] = adjusted
                    if hasattr(t, "mean_history") and len(t.mean_history) > 0:
                        t.mean_history[-1][:2] = adjusted
                    # Reflect to tlwh (keep size; shift position)
                    x, y, w, h = t.tlwh
                    new_x = adjusted[0] - 0.5*w
                    new_y = adjusted[1] - 0.5*h
                    # t.tlwh = np.array([new_x, new_y, w, h], dtype=float)
                    t.mean[0:2] = np.array([new_x, new_y], dtype=float)

    @staticmethod
    def compute_theta(roi):
        """
        Computes the opening angle theta of the direction cone from four ROI points.
        ROI assumed to be ordered as [(E1), (E2), (O2), (O1)].
        """
        E1, E2, O2, O1 = roi
        v1 = np.array(O2) - np.array(E1)
        v2 = np.array(O1) - np.array(E2)
        dot = np.dot(v1, v2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        theta = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
        return theta
    
    @staticmethod
    def _get_center_from_tlwh(tlwh):
        x, y, w, h = tlwh
        return np.array([x + 0.5*w, y + 0.5*h], dtype=float)

    @staticmethod
    def _point_in_polygon(pt, poly):
        """Ray casting; poly shape (M,2). Returns True if inside or on boundary."""
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # Check intersection with horizontal ray
            cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ( (y2 - y1) + 1e-9 ) + x1)
            if cond:
                inside = not inside
        return inside

    @staticmethod
    def _closest_point_on_segment(p, a, b):
        """Project point p to segment ab, return closest point."""
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)
        t = max(0.0, min(1.0, t))
        return a + t * ab

    @classmethod
    def _clamp_point_to_polygon(cls, pt, poly):
        """Clamp point to nearest point on polygon boundary."""
        best = None
        best_d2 = 1e18
        n = len(poly)
        for i in range(n):
            a = poly[i].astype(float)
            b = poly[(i + 1) % n].astype(float)
            q = cls._closest_point_on_segment(pt, a, b)
            d2 = np.sum((q - pt)**2)
            if d2 < best_d2:
                best_d2 = d2
                best = q
        return best if best is not None else pt

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-9)

    @staticmethod
    def _angle_of(vec):
        """Angle of vector in radians (−pi, pi]."""
        return math.atan2(vec[1], vec[0])

    @staticmethod
    def _angle_diff(a, b):
        """Smallest signed angle a−b in radians (−pi, pi]."""
        d = (a - b + math.pi) % (2*math.pi) - math.pi
        return d

    @staticmethod
    def _cone_axis_and_theta(roi):
        """From four ROI points [(E1),(E2),(O2),(O1)] get cone axis unit vector and theta (degrees)."""
        E1, E2, O2, O1 = roi
        v1 = Fasttracker._normalize(np.array(O2) - np.array(E1))
        v2 = Fasttracker._normalize(np.array(O1) - np.array(E2))
        axis = Fasttracker._normalize(v1 + v2)  # average direction
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        theta = math.degrees(math.acos(dot))    # opening angle
        return axis, theta

    def _clamp_to_cone(self, anchor_pt, curr_pt, axis_u, theta_deg):
        """
        Enforce a direction cone centered on 'axis_u' with opening angle 'theta_deg'.
        If the displacement delta = curr_pt - anchor_pt deviates beyond theta/2 from axis_u,
        clamp delta to the nearest cone boundary while preserving its magnitude.
        Returns the adjusted current point.
        """
        delta = np.asarray(curr_pt, dtype=float) - np.asarray(anchor_pt, dtype=float)
        mag = np.linalg.norm(delta)
        if mag < 3.0: # configurable
            return curr_pt  # no displacement, nothing to adjust

        # Normalize vectors
        delta_u = delta / mag
        axis_u  = self._normalize(np.asarray(axis_u, dtype=float))

        # Angle between delta and axis_u (0..pi)
        cosang = float(np.clip(np.dot(delta_u, axis_u), -1.0, 1.0))
        ang = math.acos(cosang)
        half = math.radians(theta_deg) * 0.5

        if ang <= half:
            # Already within cone
            return curr_pt

        # Determine which side to clamp to (sign via 2D cross product z-component)
        cross_z = axis_u[0] * delta_u[1] - axis_u[1] * delta_u[0]
        sign = 1.0 if cross_z > 0 else -1.0  # +half on left side, -half on right side

        # Build the boundary direction by rotating axis_u by -+half
        c, s = math.cos(sign * half), math.sin(sign * half)

        # Rotation matrix R(theta/2) = [ [c -s], [s  c] ]
        # rotate axis_u by this matrix to get boundary direction
        boundary_dir = np.array([axis_u[0]*c - axis_u[1]*s, axis_u[0]*s + axis_u[1]*c], dtype=float)

        # Preserve the magnitude of delta
        # New point = pk_N + boundary_dir * |mag|
        clamped = np.asarray(anchor_pt, dtype=float) + boundary_dir * mag
        return clamped

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
