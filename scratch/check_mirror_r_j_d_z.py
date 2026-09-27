import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

# Let's check classes_ in model
print("Model classes:", model.classes_)

# Let's check what R (17) is predicted as
idxs_17 = [i for i, lbl in enumerate(labels) if str(lbl) == '17']
X_17 = np.asarray([data[i] for i in idxs_17], dtype=np.float32)
preds_17 = model.predict(X_17)
probas_17 = model.predict_proba(X_17)
print("Class 17 (R) self-predictions on raw dataset:")
print(f"  Acc: {np.mean(preds_17 == '17')*100:.1f}%, Mean conf: {np.mean(np.max(probas_17, axis=1))*100:.1f}%")

# What about mirrored R?
# What happens if a user signs R with their other hand? (1.0 - xs)
X_17_mirrored = []
for sample in [data[i] for i in idxs_17]:
    xs = np.array(sample[0::2])
    ys = np.array(sample[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    feat = []
    for x, y in zip((xs_m - min_x)/scale, (ys - min_y)/scale):
        feat.extend([float(x), float(y)])
    X_17_mirrored.append(feat)

X_17_m = np.asarray(X_17_mirrored, dtype=np.float32)
preds_17_m = model.predict(X_17_m)
probas_17_m = model.predict_proba(X_17_m)
unique_preds = {}
for p in preds_17_m:
    unique_preds[p] = unique_preds.get(p, 0) + 1
print(f"Class 17 (R) predictions when MIRRORED (other hand): {unique_preds}")

# What about mirrored J (9)?
idxs_9 = [i for i, lbl in enumerate(labels) if str(lbl) == '9']
X_9_mirrored = []
for sample in [data[i] for i in idxs_9]:
    xs = np.array(sample[0::2])
    ys = np.array(sample[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    feat = []
    for x, y in zip((xs_m - min_x)/scale, (ys - min_y)/scale):
        feat.extend([float(x), float(y)])
    X_9_mirrored.append(feat)
X_9_m = np.asarray(X_9_mirrored, dtype=np.float32)
preds_9_m = model.predict(X_9_m)
unique_preds_9 = {}
for p in preds_9_m:
    unique_preds_9[p] = unique_preds_9.get(p, 0) + 1
print(f"Class 9 (J) predictions when MIRRORED (other hand): {unique_preds_9}")

# What about mirrored D (3) and Z (25)?
idxs_3 = [i for i, lbl in enumerate(labels) if str(lbl) == '3']
X_3_mirrored = []
for sample in [data[i] for i in idxs_3]:
    xs = np.array(sample[0::2])
    ys = np.array(sample[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    feat = []
    for x, y in zip((xs_m - min_x)/scale, (ys - min_y)/scale):
        feat.extend([float(x), float(y)])
    X_3_mirrored.append(feat)
X_3_m = np.asarray(X_3_mirrored, dtype=np.float32)
preds_3_m = model.predict(X_3_m)
unique_preds_3 = {}
for p in preds_3_m:
    unique_preds_3[p] = unique_preds_3.get(p, 0) + 1
print(f"Class 3 (D) predictions when MIRRORED (other hand): {unique_preds_3}")

idxs_25 = [i for i, lbl in enumerate(labels) if str(lbl) == '25']
X_25_mirrored = []
for sample in [data[i] for i in idxs_25]:
    xs = np.array(sample[0::2])
    ys = np.array(sample[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    feat = []
    for x, y in zip((xs_m - min_x)/scale, (ys - min_y)/scale):
        feat.extend([float(x), float(y)])
    X_25_mirrored.append(feat)
X_25_m = np.asarray(X_25_mirrored, dtype=np.float32)
preds_25_m = model.predict(X_25_m)
unique_preds_25 = {}
for p in preds_25_m:
    unique_preds_25[p] = unique_preds_25.get(p, 0) + 1
print(f"Class 25 (Z) predictions when MIRRORED (other hand): {unique_preds_25}")
