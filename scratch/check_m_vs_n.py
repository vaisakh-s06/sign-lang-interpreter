import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

for target_lbl, target_name in [(12, 'M'), (13, 'N'), (18, 'S'), (19, 'T'), (0, 'A')]:
    mask = (labels == target_lbl)
    samples = data[mask]
    
    # Thumb tip: landmark 4 (x: 8, y: 9)
    # Index mcp: 5 (x: 10, y: 11), pip: 6 (x: 12, y: 13), tip: 8 (x: 16, y: 17)
    # Middle mcp: 9 (x: 18, y: 19), pip: 10 (x: 20, y: 21), tip: 12 (x: 24, y: 25)
    # Ring mcp: 13 (x: 26, y: 27), pip: 14 (x: 28, y: 29), tip: 16 (x: 32, y: 33)
    # Pinky mcp: 17 (x: 34, y: 35), pip: 18 (x: 36, y: 37), tip: 20 (x: 40, y: 41)
    
    thumb_x = samples[:, 8]
    thumb_y = samples[:, 9]
    mid_mcp_x = samples[:, 18]
    ring_mcp_x = samples[:, 26]
    pinky_mcp_x = samples[:, 34]
    
    print(f"Class {target_name} ({target_lbl}): N={len(samples)}")
    print(f"  Thumb tip x mean: {np.mean(thumb_x):.4f}")
    print(f"  Middle MCP x mean: {np.mean(mid_mcp_x):.4f}")
    print(f"  Ring MCP x mean: {np.mean(ring_mcp_x):.4f}")
    print(f"  Pinky MCP x mean: {np.mean(pinky_mcp_x):.4f}")
    # Relative thumb position along palm width
    # In M, thumb is positioned between ring and pinky (thumb_x > ring_mcp_x)
    # In N, thumb is positioned between middle and ring (thumb_x between mid_mcp_x and ring_mcp_x)
    # In T, thumb is positioned between index and middle
    print(f"  Thumb x relative to Ring MCP (thumb_x - ring_x): {np.mean(thumb_x - ring_mcp_x):.4f}")
    print(f"  Thumb x relative to Mid MCP (thumb_x - mid_x): {np.mean(thumb_x - mid_mcp_x):.4f}")
