import os, pickle, math, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading raw scale-normalized data from data.pickle...")
with open('./data.pickle', 'rb') as f:
    d = pickle.load(f)

raw_data = d['data']
raw_labels = d['labels']

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

# Tilt angles for natural hand gestures
angles = [-20, -15, -10, -5, 5, 10, 15, 20]

np.random.seed(42)

for sample, label in zip(raw_data, raw_labels):
    if len(sample) != 42:
        continue
    xs = np.array(sample[0::2], dtype=np.float32)
    ys = np.array(sample[1::2], dtype=np.float32)
    c_int = int(str(label))

    # 1. Base normalized sample
    base_feat = normalize_coords(xs, ys)
    aug_data.append(base_feat)
    aug_labels.append(label)

    # 2. Horizontal mirror
    # NOTE: For R (17) and U (20), mirroring inverts the crossing!
    # If we mirror R, its crossed index/mid becomes uncrossed, causing confusion with U.
    # So we do NOT mirror R and U horizontally across x without preserving chirality,
    # OR we only mirror non-chiral signs.
    xs_m = 1.0 - xs
    if c_int not in [17, 20]:
        aug_data.append(normalize_coords(xs_m, ys))
        aug_labels.append(label)

    # 3. Rotations on base
    for angle in angles:
        xr, yr = rotate_coords(xs, ys, angle)
        aug_data.append(normalize_coords(xr, yr))
        aug_labels.append(label)

        if c_int not in [17, 20]:
            xr_m, yr_m = rotate_coords(xs_m, ys, angle)
            aug_data.append(normalize_coords(xr_m, yr_m))
            aug_labels.append(label)

    # 4. Realistic landmark jitter / variation
    # Simulates slight differences in finger thickness, curling, and distance
    for _ in range(4):
        noise_x = np.random.normal(0, 0.02, size=xs.shape).astype(np.float32)
        noise_y = np.random.normal(0, 0.02, size=ys.shape).astype(np.float32)
        # Keep wrist anchored
        noise_x[0] = 0
        noise_y[0] = 0
        xs_j = xs + noise_x
        ys_j = ys + noise_y
        aug_data.append(normalize_coords(xs_j, ys_j))
        aug_labels.append(label)

print(f"Total augmented dataset: {len(aug_data)} samples.")

X = np.asarray(aug_data, dtype=np.float32)
y = np.asarray(aug_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=True, stratify=y, random_state=42
)

# Train a robust Random Forest with 250 trees
clf = RandomForestClassifier(
    n_estimators=250,
    max_depth=35,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training robust RandomForestClassifier on enriched dataset...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Overall Test Accuracy: {acc * 100:.2f}%")

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': clf}, f)
print("Saved clean, robust model to model.p successfully!")
