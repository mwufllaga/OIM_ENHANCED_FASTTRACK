#!/usr/bin/env python3
"""
Standalone script to reproduce ReID comparison at frame 274.

Uses the CORRECT approach: pass full-size original frame + bbox to
extract_query_feature (OIMNetPlus uses Faster R-CNN backbone → needs
full image context for RoI pooling).

Usage:
    cd FastTracker
    conda run -n FastTracker python tools/test_reid.py
"""
import sys
import os

# Add project root to path so we can import yolox
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from yolox.reid import extract_query_feature

# ── Configuration ─────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "debug_0_320.mp4")

# MOT result file (frame_id, track_id, x, y, w, h, conf, -1, -1, -1)
RESULT_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "YOLOX_outputs/yolox_x_mix_mot20_ch/track_vis/2026_03_04_12_26_00.txt"
)

# Which frames and tracks to extract
# T20 reference: frame 73 (tracklet_len=5 trigger)
# T28 references: frames 221, 226, 231 (tracklet_len 5/10/15 triggers)
# Current (comparison): frame 274 for both T20, T28
QUERIES = {
    "ref_T20_f73":    {"frame": 73,  "track_id": 20},
    "ref_T28_f221":   {"frame": 221, "track_id": 28},
    "ref_T28_f226":   {"frame": 226, "track_id": 28},
    "ref_T28_f231":   {"frame": 231, "track_id": 28},
    "cur_T20_f274":   {"frame": 274, "track_id": 20},
    "cur_T28_f274":   {"frame": 274, "track_id": 28},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_mot_results(result_path: str) -> dict:
    """Parse MOT result file → {(frame_id, track_id): (x, y, w, h)}."""
    bboxes = {}
    with open(result_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            bboxes[(fid, tid)] = (x, y, w, h)
    return bboxes


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a specific frame (0-indexed) from video, return BGR numpy array."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Failed to read frame {frame_idx} from {video_path}"
    return frame


def extract_embedding(rgb_img: np.ndarray, xywh: tuple) -> np.ndarray:
    """
    Extract 256-dim L2-normalized embedding from full-size RGB image + bbox.
    Same logic as Fasttracker._extract_embedding().
    """
    h, w = rgb_img.shape[:2]
    x, y, bw, bh = xywh
    # xywh → tlbr → normalized [0,1]
    bbox_norm = {
        "x1": max(0.0, x / w),
        "y1": max(0.0, y / h),
        "x2": min(1.0, (x + bw) / w),
        "y2": min(1.0, (y + bh) / h),
    }
    print(f"    bbox_norm: x1={bbox_norm['x1']:.4f} y1={bbox_norm['y1']:.4f} "
          f"x2={bbox_norm['x2']:.4f} y2={bbox_norm['y2']:.4f}")

    feat = extract_query_feature(rgb_img, bbox_norm)
    assert feat is not None, "Feature extraction failed"
    emb = np.array(feat, dtype=np.float32)
    print(f"    shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    return emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (dot product for L2-normalized vectors)."""
    return float(np.dot(a, b))


def save_debug_crop(bgr_frame: np.ndarray, xywh: tuple, name: str, out_dir: str):
    """Save a crop of the bbox region from the full frame, with bbox rectangle drawn."""
    h, w = bgr_frame.shape[:2]
    x, y, bw, bh = xywh
    x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)

    # Expand crop region with generous padding for context
    pad = max(bw, bh) * 2
    cx1 = max(0, int(x1 - pad))
    cy1 = max(0, int(y1 - pad))
    cx2 = min(w, int(x2 + pad))
    cy2 = min(h, int(y2 + pad))

    crop = bgr_frame[cy1:cy2, cx1:cx2].copy()
    # Draw bbox on crop (offset by crop origin)
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2)
    cv2.putText(crop, name, (x1 - cx1, y1 - cy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.jpg")
    cv2.imwrite(path, crop)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ReID Reproduction Script — Frame 274")
    print("Full-image + bbox mode (same as tracker)")
    print("=" * 60)

    video_path = os.path.abspath(VIDEO_PATH)
    result_path = os.path.abspath(RESULT_PATH)
    print(f"\nVideo:   {video_path}")
    print(f"Results: {result_path}")
    assert os.path.isfile(video_path), f"Video not found: {video_path}"
    assert os.path.isfile(result_path), f"Results not found: {result_path}"

    # 1) Parse MOT result file for bbox coordinates
    print("\n[1] Parsing MOT result file ...")
    bboxes = parse_mot_results(result_path)
    print(f"    Loaded {len(bboxes)} entries")

    # Verify all required (frame, track) pairs exist
    for name, q in QUERIES.items():
        key = (q["frame"], q["track_id"])
        assert key in bboxes, f"Missing bbox for {name}: frame={q['frame']}, track={q['track_id']}"
        xywh = bboxes[key]
        print(f"    {name}: xywh=({xywh[0]:.1f}, {xywh[1]:.1f}, {xywh[2]:.1f}, {xywh[3]:.1f})")

    # 2) Collect unique frames to read
    frames_needed = sorted(set(q["frame"] for q in QUERIES.values()))
    print(f"\n[2] Reading {len(frames_needed)} frames from video: {frames_needed}")
    frame_cache = {}      # fid → RGB
    frame_cache_bgr = {}  # fid → BGR (for debug images)
    for fid in frames_needed:
        bgr = read_frame(video_path, fid)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame_cache[fid] = rgb
        frame_cache_bgr[fid] = bgr
        print(f"    Frame {fid}: {rgb.shape}")

    # 2.5) Save debug crops with bbox drawn
    debug_dir = os.path.join(os.path.dirname(__file__), "..", "reid_debug_bbox")
    print(f"\n[2.5] Saving debug crops with bbox to {os.path.abspath(debug_dir)} ...")
    for name, q in QUERIES.items():
        bgr = frame_cache_bgr[q["frame"]]
        xywh = bboxes[(q["frame"], q["track_id"])]
        save_debug_crop(bgr, xywh, name, debug_dir)

    # 3) Extract embeddings
    print("\n[3] Extracting embeddings (full image + bbox) ...")
    embeddings = {}
    for name, q in QUERIES.items():
        print(f"  {name}:")
        rgb = frame_cache[q["frame"]]
        xywh = bboxes[(q["frame"], q["track_id"])]
        embeddings[name] = extract_embedding(rgb, xywh)

    # 4) Compute cross-similarity
    emb_ref_t20 = embeddings["ref_T20_f73"]
    # Average T28 refs (same as tracker compares with mean of ref_embeddings)
    emb_ref_t28 = np.mean([
        embeddings["ref_T28_f221"],
        embeddings["ref_T28_f226"],
        embeddings["ref_T28_f231"],
    ], axis=0)
    emb_cur_t20 = embeddings["cur_T20_f274"]
    emb_cur_t28 = embeddings["cur_T28_f274"]

    print("\n[4] Similarity matrix:")
    sim_t20_cur_vs_t20_ref = cosine_sim(emb_cur_t20, emb_ref_t20)
    sim_t20_cur_vs_t28_ref = cosine_sim(emb_cur_t20, emb_ref_t28)
    sim_t28_cur_vs_t20_ref = cosine_sim(emb_cur_t28, emb_ref_t20)
    sim_t28_cur_vs_t28_ref = cosine_sim(emb_cur_t28, emb_ref_t28)

    print(f"  T20_current vs T20_ref  = {sim_t20_cur_vs_t20_ref:.4f}")
    print(f"  T20_current vs T28_ref  = {sim_t20_cur_vs_t28_ref:.4f}")
    print(f"  T28_current vs T20_ref  = {sim_t28_cur_vs_t20_ref:.4f}")
    print(f"  T28_current vs T28_ref  = {sim_t28_cur_vs_t28_ref:.4f}")

    # 5) Decision (same logic as _check_separations)
    score_correct = sim_t20_cur_vs_t20_ref + sim_t28_cur_vs_t28_ref
    score_swapped = sim_t20_cur_vs_t28_ref + sim_t28_cur_vs_t20_ref

    print(f"\n[5] Decision:")
    print(f"  score_correct (keep IDs)  = {score_correct:.4f}")
    print(f"  score_swapped (swap IDs)  = {score_swapped:.4f}")

    if score_swapped > score_correct:
        print("  → SWAP detected! IDs should be swapped.")
    else:
        print("  → IDs are correct, no swap needed.")

    # 6) Per-ref T28 detail
    print(f"\n[6] Per-ref T28 similarities (detail):")
    for ref_name in ["ref_T28_f221", "ref_T28_f226", "ref_T28_f231"]:
        emb = embeddings[ref_name]
        s1 = cosine_sim(emb_cur_t20, emb)
        s2 = cosine_sim(emb_cur_t28, emb)
        print(f"  {ref_name}: T20_cur→{s1:.4f}  T28_cur→{s2:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
