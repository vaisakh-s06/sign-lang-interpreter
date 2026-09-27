import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

idxs_9 = [i for i, lbl in enumerate(labels) if str(lbl) == '9']
idxs_8 = [i for i, lbl in enumerate(labels) if str(lbl) == '8']
idxs_24 = [i for i, lbl in enumerate(labels) if str(lbl) == '24']

print("Class 9 (J) samples:", len(idxs_9))
print("Class 8 (I) samples:", len(idxs_8))
print("Class 24 (Y) samples:", len(idxs_24))

# Let's inspect landmark relations in J (9) vs I (8) vs Y (24)
for name, idxs in [('I (8)', idxs_8), ('J (9)', idxs_9), ('Y (24)', idxs_24)]:
    th_y_list = []
    pky_x_list = []
    pky_y_list = []
    th_pky_dist = []
    for i in idxs:
        s = data[i]
        xs, ys = s[0::2], s[1::2]
        # Thumb tip = 4, Pinky tip = 20
        th_y_list.append(ys[4])
        pky_x_list.append(xs[20] - xs[17]) # pinky lean
        pky_y_list.append(ys[20])
        th_pky_dist.append(np.hypot(xs[4] - xs[20], ys[4] - ys[20]))
    print(f"\n{name}:")
    print(f"  Thumb tip y: mean={np.mean(th_y_list):.3f}")
    print(f"  Pinky tip y: mean={np.mean(pky_y_list):.3f}")
    print(f"  Pinky dx (tip - mcp): mean={np.mean(pky_x_list):.3f}")
    print(f"  Thumb-to-Pinky tip distance: mean={np.mean(th_pky_dist):.3f}")
