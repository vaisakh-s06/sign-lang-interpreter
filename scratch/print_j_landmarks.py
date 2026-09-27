import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

idxs_9 = [i for i, lbl in enumerate(d['labels']) if str(lbl) == '9']
sample = d['data'][idxs_9[0]]
xs = sample[0::2]
ys = sample[1::2]

# Wrist = 0
# Thumb: 1, 2, 3, 4
# Index: 5, 6, 7, 8
# Middle: 9, 10, 11, 12
# Ring: 13, 14, 15, 16
# Pinky: 17, 18, 19, 20

print("Hand landmark normalized coords in class 9 (J):")
names = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
         'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
         'MID_MCP', 'MID_PIP', 'MID_DIP', 'MID_TIP',
         'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
         'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

for idx, name in enumerate(names):
    print(f"  {idx:>2}: {name:<12} -> ({xs[idx]:.3f}, {ys[idx]:.3f})")
