import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

def mirror_features(feat):
    xs = np.array(feat[0::2])
    ys = np.array(feat[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    s = max(max_x - min_x, max_y - min_y)
    if s < 1e-4:
        s = 1.0
    out = []
    for x, y in zip((xs_m - min_x) / s, (ys - min_y) / s):
        out.extend([float(x), float(y)])
    return out

# For every sample, simulate left hand by mirroring, then see if canonical selection restores the exact same features
for target_lbl, name in [(10, 'K'), (12, 'M'), (13, 'N'), (17, 'R'), (3, 'D')]:
    mask = (labels == target_lbl)
    sample = data[mask][0]
    
    # Simulate user showing left hand:
    left_hand_input = mirror_features(sample)
    
    # Dual chirality:
    p_orig = model.predict_proba(np.array(left_hand_input).reshape(1, -1))[0]
    p_mirr = model.predict_proba(np.array(mirror_features(left_hand_input)).reshape(1, -1))[0]
    
    chosen_hand = "Left (mirrored)" if np.max(p_mirr) > np.max(p_orig) else "Right (direct)"
    max_p = max(np.max(p_orig), np.max(p_mirr))
    best_class = int(model.classes_[np.argmax(np.maximum(p_orig, p_mirr))])
    
    print(f"Sign {name}: Detected as {chosen_hand}, Pred={best_class} (Expected {target_lbl}), Conf={max_p:.3f}")
