import os
import cv2
import pickle
import numpy as np

# Let's inspect data.pickle for D (3), Z (25), R (17), J (9)
with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = d['data']
labels = d['labels']

print(f"Total samples: {len(data)}")

# Let's check landmark characteristics for D (3) and Z (25)
# Feature: 21 landmarks normalized
# Index tip = 8, Index MCP = 5, Index PIP = 6, Index DIP = 7
# Middle tip = 12, Middle MCP = 9
# Thumb tip = 4, Thumb MCP = 2

for target, name in [('3', 'D'), ('25', 'Z'), ('17', 'R'), ('9', 'J')]:
    idxs = [i for i, lbl in enumerate(labels) if str(lbl) == target]
    print(f"\n--- Class {target} ({name}): {len(idxs)} samples ---")
    if not idxs:
        continue
    
    idx_angles = []
    idx_slopes = []
    thumb_mid_dists = []
    cross_metrics = []
    pinky_hooks = []
    
    for i in idxs:
        feat = data[i]
        xs = feat[0::2]
        ys = feat[1::2]
        
        # Index finger slope / tilt: dx / dy
        # From MCP (5) to Tip (8)
        dx = xs[8] - xs[5]
        dy = ys[8] - ys[5]
        angle_deg = np.degrees(np.arctan2(dx, -dy)) # 0 is straight up, + is tilted right, - is tilted left
        idx_angles.append(angle_deg)
        
        # In D: thumb tip (4) touches middle tip (12) or middle PIP (10)
        d_touch = np.hypot(xs[4] - xs[12], ys[4] - ys[12])
        thumb_mid_dists.append(d_touch)
        
        # In R: index (8) crosses middle (12)
        cross = (xs[5] - xs[9]) * (xs[8] - xs[12])
        cross_metrics.append(cross)
        
        # In J: pinky hook (20) relative to MCP (17)
        p_dx = xs[20] - xs[17]
        p_dy = ys[20] - ys[17]
        pinky_hooks.append((p_dx, p_dy))
        
    print(f"  Index angle from vertical (deg): mean={np.mean(idx_angles):.2f}, min={np.min(idx_angles):.2f}, max={np.max(idx_angles):.2f}")
    print(f"  Thumb tip to Middle tip dist: mean={np.mean(thumb_mid_dists):.4f}, min={np.min(thumb_mid_dists):.4f}, max={np.max(thumb_mid_dists):.4f}")
    if target in ['17', '20']:
        print(f"  Crossing metric: mean={np.mean(cross_metrics):.5f}")
