import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

print(f"Total samples: {len(data)}")

# Landmark indices:
# Index MCP=5, PIP=6, DIP=7, TIP=8
# Middle MCP=9, PIP=10, DIP=11, TIP=12
# Ring MCP=13, PIP=14, DIP=15, TIP=16
# Pinky MCP=17, PIP=18, DIP=19, TIP=20

for target in ['3', '17', '20', '21', '10']:
    idxs = [i for i, lbl in enumerate(labels) if str(lbl) == target]
    mid_ext_list = []
    idx_ext_list = []
    mid_y_list = []
    idx_y_list = []
    for i in idxs:
        s = data[i]
        xs = s[0::2]
        ys = s[1::2]
        # Finger is extended if tip is above PIP (ys[tip] < ys[pip])
        idx_ext = ys[8] < ys[6]
        mid_ext = ys[12] < ys[10]
        idx_ext_list.append(idx_ext)
        mid_ext_list.append(mid_ext)
        idx_y_list.append(ys[8])
        mid_y_list.append(ys[12])
    print(f"Class {target:>2}: Index Extended: {np.mean(idx_ext_list)*100:.0f}%, Middle Extended: {np.mean(mid_ext_list)*100:.0f}%, Mean Index Tip Y: {np.mean(idx_y_list):.3f}, Mean Middle Tip Y: {np.mean(mid_y_list):.3f}")
