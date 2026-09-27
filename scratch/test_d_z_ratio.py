import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = d['data']
labels = d['labels']

idxs_3 = [i for i, lbl in enumerate(labels) if str(lbl) == '3'] # D
idxs_25 = [i for i, lbl in enumerate(labels) if str(lbl) == '25'] # Z

ratios_d = []
angles_d = []
for idx in idxs_3:
    s = data[idx]
    xs, ys = s[0::2], s[1::2]
    # Vector from Index MCP (5) to Index Tip (8)
    dx = xs[8] - xs[5]
    dy = ys[5] - ys[8] # positive when tip is above MCP
    ratios_d.append(dx / dy if dy != 0 else 0)
    angles_d.append(np.degrees(np.arctan2(dx, dy)))

ratios_z = []
angles_z = []
for idx in idxs_25:
    s = data[idx]
    xs, ys = s[0::2], s[1::2]
    dx = xs[8] - xs[5]
    dy = ys[5] - ys[8]
    ratios_z.append(dx / dy if dy != 0 else 0)
    angles_z.append(np.degrees(np.arctan2(dx, dy)))

print("D (Class 3):")
print(f"  Lean ratio (dx/dy): mean={np.mean(ratios_d):.4f}, min={np.min(ratios_d):.4f}, max={np.max(ratios_d):.4f}")
print(f"  Angle from vertical (deg): mean={np.mean(angles_d):.2f}, min={np.min(angles_d):.2f}, max={np.max(angles_d):.2f}")

print("\nZ (Class 25):")
print(f"  Lean ratio (dx/dy): mean={np.mean(ratios_z):.4f}, min={np.min(ratios_z):.4f}, max={np.max(ratios_z):.4f}")
print(f"  Angle from vertical (deg): mean={np.mean(angles_z):.2f}, min={np.min(angles_z):.2f}, max={np.max(angles_z):.2f}")
