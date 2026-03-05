#!/usr/bin/env python3
"""Compare ref_T20_f73 vs cur_T28_f274 manually — 600x600 crop mode."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np
from yolox.reid import extract_query_feature

VIDEO = "/Users/liaoruoxing/Documents/program/mot/debug_0_320.mp4"

cap = cv2.VideoCapture(VIDEO)

cap.set(cv2.CAP_PROP_POS_FRAMES, 73)
ret, f73 = cap.read()

cap.set(cv2.CAP_PROP_POS_FRAMES, 274)
ret2, f274 = cap.read()
cap.release()

h, w = f73.shape[:2]
print(f"Original image size: {w}x{h}")

CROP_SIZE = 400

def crop_around_bbox(frame, x, y, bw, bh, crop_size=CROP_SIZE):
    """Crop a crop_size x crop_size region centered on the bbox. Return (crop, new_bbox_xywh)."""
    H, W = frame.shape[:2]
    cx = x + bw / 2
    cy = y + bh / 2
    half = crop_size / 2

    # Crop bounds (clamp to image)
    cx1 = int(max(0, cx - half))
    cy1 = int(max(0, cy - half))
    cx2 = int(min(W, cx + half))
    cy2 = int(min(H, cy + half))

    crop = frame[cy1:cy2, cx1:cx2].copy()
    # Translate bbox to crop coordinates
    new_x = x - cx1
    new_y = y - cy1
    print(f"  Crop region: ({cx1},{cy1})-({cx2},{cy2}), size={cx2-cx1}x{cy2-cy1}")
    print(f"  Original bbox: ({x:.1f},{y:.1f},{bw:.1f},{bh:.1f})")
    print(f"  New bbox in crop: ({new_x:.1f},{new_y:.1f},{bw:.1f},{bh:.1f})")
    return crop, (new_x, new_y, bw, bh)

# T20 @ frame 73: xywh=(2109.78, 598.54, 25.90, 68.57)
print("\n--- T20 @ frame 73 ---")
crop1, bbox1 = crop_around_bbox(f73, 2109.78, 598.54, 25.90, 68.57)

# T28 @ frame 274: xywh=(2119.01, 667.82, 30.45, 77.02)
print("\n--- T28 @ frame 274 ---")
crop2, bbox2 = crop_around_bbox(f274, 2119.01, 667.82, 30.45, 77.02)

# Convert to normalized bbox for extract_query_feature
def to_norm_bbox(crop, xywh):
    ch, cw = crop.shape[:2]
    x, y, bw, bh = xywh
    return {"x1": x/cw, "y1": y/ch, "x2": (x+bw)/cw, "y2": (y+bh)/ch}

b1 = to_norm_bbox(crop1, bbox1)
b2 = to_norm_bbox(crop2, bbox2)
print(f"\nNorm bbox T20@f73:  {b1}")
print(f"Norm bbox T28@f274: {b2}")

# Save 600x600 crops with bbox drawn
os.makedirs("reid_debug_bbox", exist_ok=True)
for name, crop, bx in [("crop600_T20_f73", crop1, bbox1), ("crop600_T28_f274", crop2, bbox2)]:
    vis = crop.copy()
    x, y, bw, bh = bx
    cv2.rectangle(vis, (int(x), int(y)), (int(x+bw), int(y+bh)), (0, 0, 255), 2)
    cv2.putText(vis, name, (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    p = f"reid_debug_bbox/{name}.jpg"
    cv2.imwrite(p, vis)
    print(f"  Saved: {p}")

rgb1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB)
rgb2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB)

e1 = np.array(extract_query_feature(rgb1, b1), dtype=np.float32)
e2 = np.array(extract_query_feature(rgb2, b2), dtype=np.float32)

print(f"\n[400x400 crop] ref_T20_f73 vs cur_T28_f274 = {float(np.dot(e1, e2)):.4f}")

# Also test cur_T20_f274 vs ref_T20_f73 (same person?)
print("\n--- cur_T20 @ frame 274 (400x400 crop) ---")
crop3, bbox3 = crop_around_bbox(f274, 2096.18, 650.06, 28.57, 81.23)
b3 = to_norm_bbox(crop3, bbox3)
print(f"Norm bbox cur_T20@f274: {b3}")
# Save debug
vis3 = crop3.copy()
x3, y3, bw3, bh3 = bbox3
cv2.rectangle(vis3, (int(x3), int(y3)), (int(x3+bw3), int(y3+bh3)), (0, 0, 255), 2)
cv2.putText(vis3, "crop400_cur_T20_f274", (int(x3), int(y3)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.imwrite("reid_debug_bbox/crop400_cur_T20_f274.jpg", vis3)
rgb3 = cv2.cvtColor(crop3, cv2.COLOR_BGR2RGB)
e3 = np.array(extract_query_feature(rgb3, b3), dtype=np.float32)
print(f"[400x400 crop] cur_T20_f274 vs ref_T20_f73 = {float(np.dot(e3, e1)):.4f}")
print(f"[400x400 crop] cur_T20_f274 vs cur_T28_f274 = {float(np.dot(e3, e2)):.4f}")

# Also compare with full-image mode for reference
print("\n--- Full image mode (for comparison) ---")
b1_full = {"x1": 2109.78/w, "y1": 598.54/h, "x2": (2109.78+25.90)/w, "y2": (598.54+68.57)/h}
b2_full = {"x1": 2119.01/w, "y1": 667.82/h, "x2": (2119.01+30.45)/w, "y2": (667.82+77.02)/h}
e1f = np.array(extract_query_feature(cv2.cvtColor(f73, cv2.COLOR_BGR2RGB), b1_full), dtype=np.float32)
e2f = np.array(extract_query_feature(cv2.cvtColor(f274, cv2.COLOR_BGR2RGB), b2_full), dtype=np.float32)
print(f"[Full image] ref_T20_f73 vs cur_T28_f274 = {float(np.dot(e1f, e2f)):.4f}")
