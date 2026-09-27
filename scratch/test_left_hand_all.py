import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

LABELS_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'Hello', 27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.',
    33: 'lional messi-the goat🐐'
}

def mirror_sample(s):
    xs = np.array(s[0::2])
    ys = np.array(s[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    feat = []
    for x, y in zip((xs_m - min_x)/scale, (ys - min_y)/scale):
        feat.extend([float(x), float(y)])
    return feat

print("EVALUATING MODEL ON LEFT HAND (MIRRORED COORDINATES):")
unique_labels = sorted(list(set(labels)), key=lambda x: int(x) if x.isdigit() else 999)

for lbl in unique_labels:
    idxs = [i for i, l in enumerate(labels) if str(l) == str(lbl)]
    X_mirrored = np.asarray([mirror_sample(data[i]) for i in idxs], dtype=np.float32)
    preds = model.predict(X_mirrored)
    acc = np.mean(preds == str(lbl)) * 100
    char_name = LABELS_DICT.get(int(lbl), f"Sign {lbl}")
    status = "OK" if acc >= 95 else ("WARN" if acc >= 80 else "FAIL")
    if acc < 95:
        # What is it misclassified as?
        mis = {}
        for p in preds[preds != str(lbl)]:
            p_name = LABELS_DICT.get(int(p), p)
            mis[p_name] = mis.get(p_name, 0) + 1
        print(f"  Class {lbl:>2} ({char_name:<16}): {acc:.1f}% [{status}] -> Misclassified as: {mis}")
    else:
        print(f"  Class {lbl:>2} ({char_name:<16}): {acc:.1f}% [{status}]")
