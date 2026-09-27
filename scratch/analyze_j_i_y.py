import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)
classes = np.array([int(c) for c in model.classes_])

print("Analyzing J (9), I (8), Y (24) landmark metrics:")

for lbl, name in [(9, 'J'), (8, 'I'), (24, 'Y')]:
    samples = data[labels == lbl]
    xs = samples[:, 0::2]
    ys = samples[:, 1::2]
    
    # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
    # Pinky tip y vs pip y (ys[20] vs ys[18])
    # Thumb tip y vs mcp y (ys[4] vs ys[2] or ys[1])
    # Thumb extension: distance of thumb tip (4) from wrist (0) or index MCP (5)
    th_tip_x = xs[:, 4]
    th_tip_y = ys[:, 4]
    pinky_tip_x = xs[:, 20]
    pinky_tip_y = ys[:, 20]
    pinky_mcp_x = xs[:, 17]
    pinky_mcp_y = ys[:, 17]
    
    # Pinky extension (dy > 0 means pointing up)
    pinky_dy = ys[:, 17] - ys[:, 20]
    pinky_dx = xs[:, 20] - xs[:, 17] # hook metric
    
    # Thumb extension (distance from palm center / index MCP)
    th_idx_dist = np.sqrt((xs[:, 4] - xs[:, 5])**2 + (ys[:, 4] - ys[:, 5])**2)
    th_ext = abs(xs[:, 4] - xs[:, 2]) # thumb horizontal extension
    
    print(f"\n--- Class {name} ({lbl}) N={len(samples)} ---")
    print(f"  Thumb tip x: mean={np.mean(th_tip_x):.3f}, y: mean={np.mean(th_tip_y):.3f}")
    print(f"  Thumb dist from Index MCP: mean={np.mean(th_idx_dist):.3f} (min={np.min(th_idx_dist):.3f}, max={np.max(th_idx_dist):.3f})")
    print(f"  Pinky tip x: mean={np.mean(pinky_tip_x):.3f}, y: mean={np.mean(pinky_tip_y):.3f}")
    print(f"  Pinky hook (x20 - x17): mean={np.mean(pinky_dx):.3f} (min={np.min(pinky_dx):.3f}, max={np.max(pinky_dx):.3f})")
    print(f"  Pinky vertical extension (y17 - y20): mean={np.mean(pinky_dy):.3f}")
