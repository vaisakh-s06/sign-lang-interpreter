import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

def mirror_features(data_aux):
    xs = np.array(data_aux[0::2])
    ys = np.array(data_aux[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-4:
        scale = 1.0
    out = []
    for x, y in zip((xs_m - min_x) / scale, (ys - min_y) / scale):
        out.extend([float(x), float(y)])
    return out

# Let's inspect G (6) and H (7)
print("--- G vs H comparison ---")
for lbl, name in [(6, 'G'), (7, 'H')]:
    mask = (labels == lbl)
    samples = data[mask]
    xs = samples[:, 0::2]
    ys = samples[:, 1::2]
    # In H, both index (8) and middle (12) are pointing horizontally!
    # In G, only index (8) is extended horizontally! Middle (12) is curled!
    mid_ext = abs(xs[:, 12] - xs[:, 9]) # horizontal extension of middle
    idx_ext = abs(xs[:, 8] - xs[:, 5])  # horizontal extension of index
    print(f"Class {name} ({lbl}):")
    print(f"  Index horiz extension |x8 - x5|: mean={np.mean(idx_ext):.3f}")
    print(f"  Middle horiz extension |x12 - x9|: mean={np.mean(mid_ext):.3f}")
    print(f"  Middle tip x (12): mean={np.mean(xs[:, 12]):.3f}")
    print(f"  Index tip x (8): mean={np.mean(xs[:, 8]):.3f}")

# Let's inspect T (19) vs A (0), S (18), N (13), M (12)
print("\n--- Fist family: T (19) vs A (0), S (18), N (13), M (12) ---")
for lbl, name in [(19, 'T'), (0, 'A'), (18, 'S'), (13, 'N'), (12, 'M')]:
    mask = (labels == lbl)
    samples = data[mask]
    xs = samples[:, 0::2]
    ys = samples[:, 1::2]
    # Thumb tip: 4
    # Index mcp: 5, pip: 6
    # Middle mcp: 9, pip: 10
    # Ring mcp: 13, pip: 14
    # Pinky mcp: 17, pip: 18
    th_x = xs[:, 4]
    th_y = ys[:, 4]
    idx_mcp_x = xs[:, 5]
    mid_mcp_x = xs[:, 9]
    ring_mcp_x = xs[:, 13]
    pinky_mcp_x = xs[:, 17]
    
    print(f"Class {name} ({lbl}):")
    print(f"  Thumb tip (x={np.mean(th_x):.3f}, y={np.mean(th_y):.3f})")
    print(f"  Thumb x relative to Index MCP (th - idx): mean={np.mean(th_x - idx_mcp_x):.3f}")
    print(f"  Thumb x relative to Middle MCP (th - mid): mean={np.mean(th_x - mid_mcp_x):.3f}")
    print(f"  Thumb x relative to Ring MCP (th - ring): mean={np.mean(th_x - ring_mcp_x):.3f}")
