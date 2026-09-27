import os
import pickle
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading scale-normalized dataset from data.pickle...")
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

raw_data = data_dict['data']
raw_labels = data_dict['labels']
print(f"Loaded {len(raw_data)} raw samples across {len(set(raw_labels))} classes.")

def normalize_coords(xs, ys):
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-4:
        scale = 1.0
    xs_norm = (xs - min_x) / scale
    ys_norm = (ys - min_y) / scale
    feat = []
    for x, y in zip(xs_norm, ys_norm):
        feat.extend([float(x), float(y)])
    return feat

def rotate_coords(xs, ys, angle_deg):
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    cx, cy = xs[0], ys[0] # wrist
    xs_rot = (xs - cx) * cos_a - (ys - cy) * sin_a + cx
    ys_rot = (xs - cx) * sin_a + (ys - cy) * cos_a + cy
    return xs_rot, ys_rot

aug_data = []
aug_labels = []

for sample, label in zip(raw_data, raw_labels):
    if len(sample) != 42:
        continue
    xs = np.array(sample[0::2])
    ys = np.array(sample[1::2])

    # 1. Base normalized sample
    aug_data.append(normalize_coords(xs, ys))
    aug_labels.append(label)

    # 2. Mirror horizontally (1.0 - xs)
    xs_m = 1.0 - xs
    aug_data.append(normalize_coords(xs_m, ys))
    aug_labels.append(label)

    # 3. Small rotations for tilt invariance (±5°, ±10°)
    for angle in [-10, -5, 5, 10]:
        xr, yr = rotate_coords(xs, ys, angle)
        aug_data.append(normalize_coords(xr, yr))
        aug_labels.append(label)

        xr_m, yr_m = rotate_coords(xs_m, ys, angle)
        aug_data.append(normalize_coords(xr_m, yr_m))
        aug_labels.append(label)

print(f"Total augmented dataset: {len(aug_data)} samples.")

X = np.asarray(aug_data, dtype=np.float32)
y = np.asarray(aug_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=True, stratify=y, random_state=42
)

# RandomForest with balanced class weights so every sign is well-represented
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=35,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training balanced RandomForestClassifier on augmented dataset...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n==========================================")
print(f"Overall Test Accuracy: {acc * 100:.2f}%")
print(f"==========================================")

# Letter-specific checks
labels_map = {'0': 'A', '3': 'D', '4': 'E', '18': 'S'}
for c, name in labels_map.items():
    mask = (y_test == c)
    if mask.sum() > 0:
        sub_acc = accuracy_score(y_test[mask], clf.predict(X_test[mask]))
        print(f"Letter {name} (Class {c}) Test Accuracy: {sub_acc * 100:.2f}% ({mask.sum()} test samples)")

with open('model.p', 'wb') as f:
    pickle.dump({'model': clf}, f)
print("\nSaved robust model to model.p successfully!")
