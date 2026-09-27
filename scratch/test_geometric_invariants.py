import pickle
import numpy as np
import math

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = d['data']
labels = d['labels']

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

# Analyze geometric properties for 6 (G), 9 (J), 12 (M), 13 (N), 17 (R)
# Feature layout: 42 elements -> 21 (x, y) normalized pairs
# 0: wrist
# 4: thumb tip
# 5: index mcp, 6: index pip, 8: index tip
# 9: middle mcp, 10: middle pip, 12: middle tip
# 13: ring mcp, 14: ring pip, 16: ring tip
# 17: pinky mcp, 18: pinky pip, 20: pinky tip

def get_pts(feat):
    xs = feat[0::2]
    ys = feat[1::2]
    return xs, ys

print("--- GEOMETRIC INVARIANTS ANALYSIS ---")

# 1. R (17) vs U (20): Crossing index and middle fingers
# Knuckles: index MCP (5) vs middle MCP (9)
# Tips: index tip (8) vs middle tip (12)
for cls in ['17', '20']:
    indices = [i for i, lbl in enumerate(labels) if str(lbl) == cls]
    cross_vals = []
    for idx in indices:
        xs, ys = get_pts(data[idx])
        knuckle_dx = xs[5] - xs[9]
        tip_dx = xs[8] - xs[12]
        cross_vals.append(knuckle_dx * tip_dx)
    cross_vals = np.array(cross_vals)
    print(f"Class {cls} (R if 17, U if 20): knuckle_dx * tip_dx mean={np.mean(cross_vals):.5f}, min={np.min(cross_vals):.5f}, max={np.max(cross_vals):.5f}, negative_count={np.sum(cross_vals < 0)}/{len(indices)}")

# 2. M (12) vs N (13): Ring tip height vs Middle tip height
for cls in ['12', '13']:
    indices = [i for i, lbl in enumerate(labels) if str(lbl) == cls]
    ring_mid_diffs = []
    for idx in indices:
        xs, ys = get_pts(data[idx])
        diff = ys[16] - ys[12] # ring_tip.y - mid_tip.y
        ring_mid_diffs.append(diff)
    ring_mid_diffs = np.array(ring_mid_diffs)
    print(f"Class {cls} (M if 12, N if 13): ring_y - mid_y mean={np.mean(ring_mid_diffs):.4f}, min={np.min(ring_mid_diffs):.4f}, max={np.max(ring_mid_diffs):.4f}")

# 3. G (6) vs H (7): Distance between index tip and middle tip
for cls in ['6', '7']:
    indices = [i for i, lbl in enumerate(labels) if str(lbl) == cls]
    tip_dists = []
    for idx in indices:
        xs, ys = get_pts(data[idx])
        dx = xs[8] - xs[12]
        dy = ys[8] - ys[12]
        tip_dists.append(math.hypot(dx, dy))
    tip_dists = np.array(tip_dists)
    print(f"Class {cls} (G if 6, H if 7): dist(idx_tip, mid_tip) mean={np.mean(tip_dists):.4f}, min={np.min(tip_dists):.4f}, max={np.max(tip_dists):.4f}")

# 4. J (9) vs I (8): Pinky & Thumb
for cls in ['8', '9']:
    indices = [i for i, lbl in enumerate(labels) if str(lbl) == cls]
    thumb_pinky_diffs = []
    pinky_slopes = []
    for idx in indices:
        xs, ys = get_pts(data[idx])
        thumb_pinky_diffs.append(ys[4] - ys[20]) # thumb_tip.y - pinky_tip.y
        pinky_slopes.append(abs(xs[20] - xs[17]))
    print(f"Class {cls} (I if 8, J if 9): thumb_y - pinky_y mean={np.mean(thumb_pinky_diffs):.4f}, pinky_dx mean={np.mean(pinky_slopes):.4f}")
