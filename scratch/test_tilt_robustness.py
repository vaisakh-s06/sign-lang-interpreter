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

def refine_prediction(class_id, confidence, feat, proba=None):
    xs = feat[0::2]
    ys = feat[1::2]
    
    # 1. R (17) vs U (20) / V (21)
    knuckle_dx = xs[5] - xs[9]
    tip_dx = xs[8] - xs[12]
    cross_metric = knuckle_dx * tip_dx
    
    if cross_metric < -0.0003:
        if class_id in [17, 20, 21] or confidence < 0.70:
            return 17, max(confidence, 0.90)
    elif cross_metric > 0.002:
        if class_id == 17:
            u_v_dist = abs(xs[8] - xs[12])
            new_id = 21 if u_v_dist > 0.15 else 20
            return new_id, max(confidence, 0.85)

    # 2. M (12) vs N (13) vs fist letters (A:0, S:18, T:19)
    ring_mid_diff = ys[16] - ys[12]
    if class_id in [12, 13]:
        if ring_mid_diff < 0.15:
            return 12, max(confidence, 0.88)
        else:
            return 13, max(confidence, 0.88)
    elif class_id in [0, 18, 19] and proba is not None:
        p_m = proba[int(np.where(model.classes_ == '12')[0][0])] if '12' in model.classes_ else 0
        p_n = proba[int(np.where(model.classes_ == '13')[0][0])] if '13' in model.classes_ else 0
        if max(p_m, p_n) > 0.15:
            if ring_mid_diff < 0.15 and p_m >= p_n:
                return 12, max(confidence, p_m, 0.80)
            elif ring_mid_diff >= 0.15 and p_n > p_m:
                return 13, max(confidence, p_n, 0.80)

    # 3. G (6) vs H (7)
    idx_mid_dist = math.hypot(xs[8] - xs[12], ys[8] - ys[12])
    if class_id in [6, 7]:
        if idx_mid_dist > 0.40:
            return 6, max(confidence, 0.92)
        else:
            return 7, max(confidence, 0.90)

    # 4. J (9) vs I (8)
    pinky_tilt = abs(xs[20] - xs[17])
    thumb_higher = ys[4] < ys[20]
    if class_id in [8, 9]:
        if pinky_tilt > 0.18 or thumb_higher:
            return 9, max(confidence, 0.90)
        else:
            return 8, max(confidence, 0.90)

    return class_id, confidence

def rotate_and_normalize(feat, angle_deg, noise_std=0.01):
    xs = np.array(feat[0::2])
    ys = np.array(feat[1::2])
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = xs[0], ys[0]
    rx = (xs - cx) * cos_a - (ys - cy) * sin_a + cx
    ry = (xs - cx) * sin_a + (ys - cy) * cos_a + cy
    if noise_std > 0:
        rx += np.random.normal(0, noise_std, len(rx))
        ry += np.random.normal(0, noise_std, len(ry))
    min_x, max_x = np.min(rx), np.max(rx)
    min_y, max_y = np.min(ry), np.max(ry)
    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-4:
        scale = 1.0
    out = []
    for x, y in zip((rx - min_x)/scale, (ry - min_y)/scale):
        out.extend([float(x), float(y)])
    return out

angles = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
target_letters = [('6', 'G'), ('9', 'J'), ('12', 'M'), ('13', 'N'), ('17', 'R')]

print("Testing robustness across +/- 25 deg tilt angles with landmark jitter:")
for t, name in target_letters:
    indices = [i for i, lbl in enumerate(labels) if str(lbl) == t]
    total_trials = 0
    correct_trials = 0
    for ang in angles:
        for idx in indices:
            aug = rotate_and_normalize(data[idx], ang, noise_std=0.01)
            feat_arr = np.asarray(aug, dtype=np.float32).reshape(1, -1)
            proba = model.predict_proba(feat_arr)[0]
            orig_id = int(str(model.classes_[np.argmax(proba)]))
            orig_conf = float(np.max(proba))
            
            ref_id, ref_conf = refine_prediction(orig_id, orig_conf, aug, proba)
            if ref_id == int(t):
                correct_trials += 1
            total_trials += 1
    print(f"Letter {name} (Class {t:>2}): {correct_trials}/{total_trials} ({correct_trials/total_trials*100:.2f}%)")
