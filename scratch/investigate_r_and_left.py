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

# Check samples of class 17 (R)
idxs_17 = [i for i, lbl in enumerate(labels) if str(lbl) == '17']
X_17 = np.asarray([data[i] for i in idxs_17], dtype=np.float32)
probas_17 = model.predict_proba(X_17)

# What are the top 3 predicted classes for each sample of R?
confused_with = {}
for i, prob in enumerate(probas_17):
    top3_idx = np.argsort(prob)[::-1][:3]
    top3_classes = [model.classes_[idx] for idx in top3_idx]
    top3_confs = [prob[idx] for idx in top3_idx]
    pred = top3_classes[0]
    if pred != '17':
        confused_with[pred] = confused_with.get(pred, 0) + 1

print("Direct class 17 predictions:", confused_with if confused_with else "All 100% R!")

# Now check: what happens if the user shows R with their LEFT HAND?
# Left hand is mirrored!
X_17_left = []
for i in idxs_17:
    s = data[i]
    xs = np.array(s[0::2])
    ys = np.array(s[1::2])
    # Mirror x
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    feat = []
    for x, y in zip((xs_m - min_x)/scale, (ys - min_y)/scale):
        feat.extend([float(x), float(y)])
    X_17_left.append(feat)

X_17_left = np.asarray(X_17_left, dtype=np.float32)
probas_17_left = model.predict_proba(X_17_left)
left_preds = {}
for prob in probas_17_left:
    p = model.classes_[np.argmax(prob)]
    name = LABELS_DICT.get(int(p), p)
    left_preds[name] = left_preds.get(name, 0) + 1

print("When R is shown with LEFT HAND (mirrored):", left_preds)

# Now what if fingers are partially crossed or uncrossed?
# What classes have 2 fingers up or 1 finger up?
