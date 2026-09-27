import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

print("Analyzing R (17), U (20), V (21) landmark metrics in data.pickle:")

for lbl, name in [(17, 'R'), (20, 'U'), (21, 'V')]:
    samples = data[labels == lbl]
    xs = samples[:, 0::2]
    ys = samples[:, 1::2]
    
    # Landmark 5: Index MCP, 8: Index Tip
    # Landmark 9: Middle MCP, 12: Middle Tip
    knuckle_dx = xs[:, 5] - xs[:, 9]  # Vector from middle knuckle to index knuckle
    tip_dx = xs[:, 8] - xs[:, 12]      # Vector from middle tip to index tip
    
    cross_metric = knuckle_dx * tip_dx
    tip_dist = np.abs(tip_dx)
    
    # Are fingers pointing straight up?
    # dy = knuckle_y - tip_y (should be positive and large)
    idx_dy = ys[:, 5] - ys[:, 8]
    mid_dy = ys[:, 9] - ys[:, 12]
    
    print(f"\n--- Class {name} ({lbl}) N={len(samples)} ---")
    print(f"  knuckle_dx (x5 - x9): mean={np.mean(knuckle_dx):.4f} (min={np.min(knuckle_dx):.4f}, max={np.max(knuckle_dx):.4f})")
    print(f"  tip_dx (x8 - x12):     mean={np.mean(tip_dx):.4f} (min={np.min(tip_dx):.4f}, max={np.max(tip_dx):.4f})")
    print(f"  cross_metric:          mean={np.mean(cross_metric):.5f} (min={np.min(cross_metric):.5f}, max={np.max(cross_metric):.5f})")
    print(f"  tip_dist |x8 - x12|:   mean={np.mean(tip_dist):.4f} (min={np.min(tip_dist):.4f}, max={np.max(tip_dist):.4f})")
