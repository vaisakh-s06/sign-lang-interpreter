import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
raw_labels = d['labels']
print("raw_labels[0]:", raw_labels[0], type(raw_labels[0]))
print("model.classes_[:5]:", model.classes_[:5], type(model.classes_[0]))
labels = np.array([int(x) for x in raw_labels], dtype=np.int32)
data = np.array(d['data'], dtype=np.float32)


def mirror_batch(data_arr):
    # data_arr shape: (N, 42)
    # x coordinates are at indices 0, 2, 4, ...
    # y coordinates are at indices 1, 3, 5, ...
    out = np.zeros_like(data_arr)
    xs = data_arr[:, 0::2]
    ys = data_arr[:, 1::2]
    xs_m = 1.0 - xs
    min_x = np.min(xs_m, axis=1, keepdims=True)
    max_x = np.max(xs_m, axis=1, keepdims=True)
    min_y = np.min(ys, axis=1, keepdims=True)
    max_y = np.max(ys, axis=1, keepdims=True)
    scale = np.maximum(max_x - min_x, max_y - min_y)
    scale[scale < 1e-4] = 1.0
    out[:, 0::2] = (xs_m - min_x) / scale
    out[:, 1::2] = (ys - min_y) / scale
    return out

# Batch mirror
data_mirrored = mirror_batch(data)

# Test Right Hand
p_orig = model.predict_proba(data)
p_mirr = model.predict_proba(data_mirrored)
p_comb = np.maximum(p_orig, p_mirr)
classes = np.array([int(c) for c in model.classes_])
preds_right = classes[np.argmax(p_comb, axis=1)]
acc_right = np.mean(preds_right == labels) * 100

# Test Left Hand (input is data_mirrored)
p_left_orig = model.predict_proba(data_mirrored)
p_left_mirr = model.predict_proba(mirror_batch(data_mirrored))
p_left_comb = np.maximum(p_left_orig, p_left_mirr)
preds_left = classes[np.argmax(p_left_comb, axis=1)]
acc_left = np.mean(preds_left == labels) * 100

print(f"Right Hand accuracy: {acc_right:.2f}%")
print(f"Left Hand accuracy: {acc_left:.2f}%")

# Let's inspect K (10), M (12), N (13), R (17) specifically!
for target_lbl, target_name in [(10, 'K'), (12, 'M'), (13, 'N'), (17, 'R'), (3, 'D')]:
    mask = (labels == target_lbl)
    r_acc = np.mean(preds_right[mask] == target_lbl) * 100
    l_acc = np.mean(preds_left[mask] == target_lbl) * 100
    print(f"Letter {target_name} ({target_lbl}): Right={r_acc:.1f}%, Left={l_acc:.1f}%")
