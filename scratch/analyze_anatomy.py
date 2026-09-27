import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

print(f"Total samples: {len(data)}")

def analyze_hand(s):
    xs = np.array(s[0::2])
    ys = np.array(s[1::2])
    
    # Finger extension: tip is distinctly higher than PIP and MCP
    is_idx_up = (ys[8] < ys[6]) and (ys[8] < ys[5])
    is_mid_up = (ys[12] < ys[10]) and (ys[12] < ys[9])
    is_ring_up = (ys[16] < ys[14])
    is_pinky_up = (ys[20] < ys[18])
    
    # Crossing metric for index and middle
    knuckle_dx = xs[5] - xs[9]
    tip_dx = xs[8] - xs[12]
    cross = knuckle_dx * tip_dx
    
    # Lean of index
    idx_dy = ys[5] - ys[8]
    idx_dx = xs[8] - xs[5]
    lean = idx_dx / idx_dy if idx_dy > 0.02 else 0
    
    return {
        'idx_up': is_idx_up,
        'mid_up': is_mid_up,
        'ring_up': is_ring_up,
        'pinky_up': is_pinky_up,
        'cross': cross,
        'lean': lean,
        'tip_dist': abs(tip_dx)
    }

for target, name in [('3', 'D'), ('25', 'Z'), ('17', 'R'), ('20', 'U'), ('21', 'V'), ('10', 'K'), ('12', 'M'), ('13', 'N')]:
    idxs = [i for i, lbl in enumerate(labels) if str(lbl) == target]
    idx_ups = []
    mid_ups = []
    crosses = []
    leans = []
    for i in idxs:
        info = analyze_hand(data[i])
        idx_ups.append(info['idx_up'])
        mid_ups.append(info['mid_up'])
        crosses.append(info['cross'])
        leans.append(info['lean'])
    print(f"Class {target:>2} ({name:<3}): Index Up={np.mean(idx_ups)*100:.0f}%, Mid Up={np.mean(mid_ups)*100:.0f}%, Mean Cross={np.mean(crosses):.5f}, Mean Lean={np.mean(leans):.3f}")
