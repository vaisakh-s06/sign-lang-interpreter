import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

print(f"Total samples: {len(labels)}")
print(f"Unique classes: {sorted(list(set(labels)))}")

# Examine G (6)
mask_g = (labels == 6)
samples_g = data[mask_g]
print(f"\n--- G (Class 6) Analysis (N={len(samples_g)}) ---")
# Check landmarks of G
# Landmark 0: wrist, 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
# Normalized xs and ys:
# xs: samples[:, 0::2], ys: samples[:, 1::2]
xs_g = samples_g[:, 0::2]
ys_g = samples_g[:, 1::2]

print("G finger tip y vs pip y:")
print(f"  Thumb tip y (4): {np.mean(ys_g[:, 4]):.3f}, ip y (3): {np.mean(ys_g[:, 3]):.3f}")
print(f"  Index tip y (8): {np.mean(ys_g[:, 8]):.3f}, pip y (6): {np.mean(ys_g[:, 6]):.3f}")
print(f"  Middle tip y (12): {np.mean(ys_g[:, 12]):.3f}, pip y (10): {np.mean(ys_g[:, 10]):.3f}")
print(f"  Ring tip y (16): {np.mean(ys_g[:, 16]):.3f}, pip y (14): {np.mean(ys_g[:, 14]):.3f}")
print(f"  Pinky tip y (20): {np.mean(ys_g[:, 20]):.3f}, pip y (18): {np.mean(ys_g[:, 18]):.3f}")

# Index direction:
idx_dx = xs_g[:, 8] - xs_g[:, 5]
idx_dy = ys_g[:, 8] - ys_g[:, 5]
print(f"  Index dx (8-5): {np.mean(idx_dx):.3f} (min: {np.min(idx_dx):.3f}, max: {np.max(idx_dx):.3f})")
print(f"  Index dy (8-5): {np.mean(idx_dy):.3f} (min: {np.min(idx_dy):.3f}, max: {np.max(idx_dy):.3f})")
# Middle direction:
mid_dx = xs_g[:, 12] - xs_g[:, 9]
mid_dy = ys_g[:, 12] - ys_g[:, 9]
print(f"  Middle dx (12-9): {np.mean(mid_dx):.3f}, Middle dy (12-9): {np.mean(mid_dy):.3f}")

# Examine T (19)
mask_t = (labels == 19)
samples_t = data[mask_t]
print(f"\n--- T (Class 19) Analysis (N={len(samples_t)}) ---")
xs_t = samples_t[:, 0::2]
ys_t = samples_t[:, 1::2]
print("T thumb and finger positions:")
print(f"  Thumb tip x (4): {np.mean(xs_t[:, 4]):.3f}, y (4): {np.mean(ys_t[:, 4]):.3f}")
print(f"  Index mcp x (5): {np.mean(xs_t[:, 5]):.3f}, pip x (6): {np.mean(xs_t[:, 6]):.3f}")
print(f"  Middle mcp x (9): {np.mean(xs_t[:, 9]):.3f}, pip x (10): {np.mean(xs_t[:, 10]):.3f}")
print(f"  Ring mcp x (13): {np.mean(xs_t[:, 13]):.3f}, pip x (14): {np.mean(xs_t[:, 14]):.3f}")
print(f"  Pinky mcp x (17): {np.mean(xs_t[:, 17]):.3f}, pip x (18): {np.mean(xs_t[:, 18]):.3f}")
