import pickle
import numpy as np
import math

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

LABELS_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'Hello', 27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.',
    33: 'lional messi-the goat🐐'
}

def rotate_and_normalize(feat, angle_deg):
    xs = np.array(feat[0::2])
    ys = np.array(feat[1::2])
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = xs[0], ys[0]
    rx = (xs - cx) * cos_a - (ys - cy) * sin_a + cx
    ry = (xs - cx) * sin_a + (ys - cy) * cos_a + cy
    min_x, max_x = np.min(rx), np.max(rx)
    min_y, max_y = np.min(ry), np.max(ry)
    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-4:
        scale = 1.0
    out = []
    for x, y in zip((rx - min_x)/scale, (ry - min_y)/scale):
        out.extend([float(x), float(y)])
    return out

angles = [-15, -10, -5, 0, 5, 10, 15]
unique_labels = sorted(list(set(labels)), key=lambda x: int(x) if x.isdigit() else 999)

print("TESTING RAW MODEL (NO OVERRIDES) ACROSS +/- 15 DEG TILT:")
weak_classes = []
for lbl in unique_labels:
    indices = [i for i, l in enumerate(labels) if str(l) == str(lbl)]
    total = 0
    correct = 0
    for ang in angles:
        batch = [rotate_and_normalize(data[idx], ang) for idx in indices]
        X_batch = np.asarray(batch, dtype=np.float32)
        preds = model.predict(X_batch)
        correct += np.sum(preds == str(lbl))
        total += len(indices)
    acc = correct / total * 100
    char_name = LABELS_DICT.get(int(lbl), f"Sign {lbl}")
    status = "OK" if acc >= 95 else ("WARN" if acc >= 80 else "FAIL")
    if acc < 95:
        weak_classes.append((lbl, char_name, acc))
    print(f"  Class {lbl:>2} ({char_name:<16}): {acc:.1f}% [{status}]")

print("\nWeak classes (< 95%):", weak_classes)
