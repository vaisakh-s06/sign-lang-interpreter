import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

for lbl, name in [(24, 'Y'), (9, 'J'), (8, 'I')]:
    samples = data[labels == lbl]
    xs = samples[:, 0::2]
    ys = samples[:, 1::2]
    
    # Distance between thumb tip (4) and pinky tip (20)
    th_pinky_dist = np.sqrt((xs[:, 4] - xs[:, 20])**2 + (ys[:, 4] - ys[:, 20])**2)
    # Horizontal span between thumb tip and pinky tip
    th_pinky_dx = np.abs(xs[:, 4] - xs[:, 20])
    
    # Distance between thumb tip (4) and index MCP (5)
    th_idx_dist = np.sqrt((xs[:, 4] - xs[:, 5])**2 + (ys[:, 4] - ys[:, 5])**2)
    
    # Distance between thumb tip (4) and middle MCP (9)
    th_mid_dist = np.sqrt((xs[:, 4] - xs[:, 9])**2 + (ys[:, 4] - ys[:, 9])**2)

    print(f"--- {name} ({lbl}) N={len(samples)} ---")
    print(f"  th_pinky_dx: mean={np.mean(th_pinky_dx):.3f} (min={np.min(th_pinky_dx):.3f}, max={np.max(th_pinky_dx):.3f})")
    print(f"  th_pinky_dist: mean={np.mean(th_pinky_dist):.3f} (min={np.min(th_pinky_dist):.3f}, max={np.max(th_pinky_dist):.3f})")
    print(f"  th_idx_dist: mean={np.mean(th_idx_dist):.3f} (min={np.min(th_idx_dist):.3f}, max={np.max(th_idx_dist):.3f})")
    print(f"  th_mid_dist: mean={np.mean(th_mid_dist):.3f} (min={np.min(th_mid_dist):.3f}, max={np.max(th_mid_dist):.3f})")
