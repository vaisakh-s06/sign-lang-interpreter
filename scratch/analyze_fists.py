import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

print("Fist family analysis (all normalized to right hand):")
for lbl, name in [(19, 'T'), (0, 'A'), (18, 'S'), (13, 'N'), (12, 'M'), (4, 'E')]:
    samples = data[labels == lbl]
    xs = samples[:, 0::2]
    ys = samples[:, 1::2]
    
    # Thumb tip: landmark 4 (index 4)
    # Index mcp: 5, pip: 6, dip: 7, tip: 8
    # Middle mcp: 9, pip: 10, dip: 11, tip: 12
    # Ring mcp: 13, pip: 14, dip: 15, tip: 16
    # Pinky mcp: 17, pip: 18, dip: 19, tip: 20
    
    # Thumb tip position:
    th_x = xs[:, 4]
    th_y = ys[:, 4]
    
    # Distance of thumb tip to index PIP (6) and middle PIP (10)
    dist_th_idx_pip = np.sqrt((xs[:, 4] - xs[:, 6])**2 + (ys[:, 4] - ys[:, 6])**2)
    dist_th_mid_pip = np.sqrt((xs[:, 4] - xs[:, 10])**2 + (ys[:, 4] - ys[:, 10])**2)
    dist_th_ring_pip = np.sqrt((xs[:, 4] - xs[:, 14])**2 + (ys[:, 4] - ys[:, 14])**2)
    
    # Height of thumb tip relative to index PIP (ys[4] - ys[6])
    # smaller y means higher up!
    th_height_vs_idx = ys[:, 4] - ys[:, 6]
    
    # Horizontal position of thumb tip:
    # xs[5] is index MCP, xs[9] is middle MCP, xs[13] is ring MCP, xs[17] is pinky MCP
    # In normalized right hand: pinky is at x~0.0, ring at x~0.15, middle at x~0.30, index at x~0.47
    print(f"--- {name} ({lbl}) N={len(samples)} ---")
    print(f"  Thumb tip x: mean={np.mean(th_x):.3f} (min={np.min(th_x):.3f}, max={np.max(th_x):.3f})")
    print(f"  Thumb tip y: mean={np.mean(th_y):.3f} (min={np.min(th_y):.3f}, max={np.max(th_y):.3f})")
    print(f"  Thumb x relative to Middle MCP (th_x - mid_mcp_x): mean={np.mean(th_x - xs[:, 9]):.3f}")
    print(f"  Thumb x relative to Index MCP (th_x - idx_mcp_x): mean={np.mean(th_x - xs[:, 5]):.3f}")
    print(f"  Thumb tip y vs Index PIP y (th_y - idx_pip_y): mean={np.mean(th_height_vs_idx):.3f}")
    print(f"  Dist to Index PIP: mean={np.mean(dist_th_idx_pip):.3f}")
    print(f"  Dist to Middle PIP: mean={np.mean(dist_th_mid_pip):.3f}")
