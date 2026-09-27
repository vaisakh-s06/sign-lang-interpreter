import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

idxs_17 = [i for i, lbl in enumerate(d['labels']) if str(lbl) == '17']
sample = d['data'][idxs_17[0]]
xs = sample[0::2]
ys = sample[1::2]

print("Hand landmark normalized coords in class 17 (R):")
names = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
         'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
         'MID_MCP', 'MID_PIP', 'MID_DIP', 'MID_TIP',
         'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
         'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

for idx, name in enumerate(names):
    print(f"  {idx:>2}: {name:<12} -> ({xs[idx]:.3f}, {ys[idx]:.3f})")

# Check crossing:
print(f"Index MCP x: {xs[5]:.3f}, Mid MCP x: {xs[9]:.3f} -> MCP dx (Index - Mid) = {xs[5] - xs[9]:.3f}")
print(f"Index Tip x: {xs[8]:.3f}, Mid Tip x: {xs[12]:.3f} -> Tip dx (Index - Mid) = {xs[8] - xs[12]:.3f}")
