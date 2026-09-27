import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)
classes = np.array([int(c) for c in model.classes_])

def mirror_batch(data_arr):
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

data_mirr = mirror_batch(data)
probs_orig = model.predict_proba(data)
probs_mirr = model.predict_proba(data_mirr)

def refined_disambiguation_j_y(class_id, confidence, data_aux):
    if len(data_aux) != 42:
        return class_id, confidence

    xs = data_aux[0::2]
    ys = data_aux[1::2]

    idx_up = ys[8] < ys[6]
    mid_up = ys[12] < ys[10]

    # --- 1. Prevent D when two fingers are pointing upwards (R vs D vs U vs V) ---
    if class_id == 3 and mid_up:
        knuckle_dx = xs[5] - xs[9]
        tip_dx = xs[8] - xs[12]
        cross_metric = knuckle_dx * tip_dx
        tip_dist = abs(xs[8] - xs[12])
        if cross_metric < -0.0001:
            return 17, max(confidence, 0.92)  # R
        else:
            return (21 if tip_dist >= 0.14 else 20), max(confidence, 0.90)  # U or V

    # --- 2. D (3) vs Z (25) ONLY (only valid when middle finger is NOT up) ---
    if class_id in [3, 25]:
        if mid_up:
            knuckle_dx = xs[5] - xs[9]
            tip_dx = xs[8] - xs[12]
            cross_metric = knuckle_dx * tip_dx
            tip_dist = abs(xs[8] - xs[12])
            if cross_metric < -0.0001:
                return 17, max(confidence, 0.90)
            else:
                return (21 if tip_dist >= 0.14 else 20), max(confidence, 0.90)
        dx = xs[8] - xs[5]
        dy = ys[5] - ys[8]
        if dy > 0.05:
            lean = dx / dy
            if lean < -0.22:
                return 25, max(confidence, 0.90)
            elif lean >= -0.20:
                return 3, max(confidence, 0.90)

    # --- 3. R (17) vs U (20)/V (21) STRICT DISAMBIGUATION ---
    elif class_id in [17, 20, 21]:
        knuckle_dx = xs[5] - xs[9]
        tip_dx = xs[8] - xs[12]
        cross_metric = knuckle_dx * tip_dx
        tip_dist = abs(xs[8] - xs[12])

        if cross_metric < -0.0001:
            return 17, max(confidence, 0.92)  # R: twisted/crossed
        else:
            if tip_dist >= 0.14:
                return 21, max(confidence, 0.92)  # V: straight and spread
            else:
                return 20, max(confidence, 0.92)  # U: straight and together

    # --- 4. J (9) vs I (8) vs Y (24) STRICT THUMB DISAMBIGUATION ---
    # In Y: BOTH thumb and pinky MUST be extended (th_pinky_dist >= 0.82 and th_idx_dist >= 0.34).
    # In J and I: thumb is NOT extended (th_pinky_dist < 0.80).
    # If thumb is not extended, sign can NEVER be Y!
    elif class_id in [8, 9, 24]:
        pinky_up = ys[20] < ys[18]
        other_curled = (ys[8] > ys[5] - 0.05) and (ys[12] > ys[9] - 0.05) and (ys[16] > ys[13] - 0.05)
        if pinky_up and other_curled:
            th_pinky_dist = np.sqrt((xs[4] - xs[20])**2 + (ys[4] - ys[20])**2)
            th_idx_dist = np.sqrt((xs[4] - xs[5])**2 + (ys[4] - ys[5])**2)
            th_is_extended = (th_pinky_dist >= 0.82) and (th_idx_dist >= 0.34)

            # If class is 24 (Y) but thumb is NOT extended: user is signing J (or I)!
            if not th_is_extended:
                pinky_dx = xs[20] - xs[17]
                thumb_higher = ys[4] < ys[20] + 0.05
                if abs(pinky_dx) > 0.16 or thumb_higher or class_id in [9, 24]:
                    return 9, max(confidence, 0.92)  # J: hooked/tilted pinky, thumb NOT extended
                else:
                    return 8, max(confidence, 0.92)  # I: straight upright pinky
            else:
                # Thumb IS extended outwards: Y (24)
                return 24, max(confidence, 0.92)  # Y: both thumb and pinky extended

    # --- 5. G (6) vs H (7) stabilization ---
    elif class_id in [6, 7]:
        idx_horiz_ext = abs(xs[8] - xs[5])
        mid_horiz_ext = abs(xs[12] - xs[9])
        if mid_horiz_ext >= 0.40 and abs(ys[8] - ys[12]) < 0.15:
            return 7, max(confidence, 0.92)
        elif idx_horiz_ext >= 0.35 and mid_horiz_ext < 0.35:
            return 6, max(confidence, 0.92)

    # --- 6. T (19) vs Fist family (A=0, S=18, N=13, M=12) stabilization ---
    elif class_id in [19, 18, 13, 12, 0]:
        fist_curled = (ys[8] > ys[5] - 0.05) and (ys[12] > ys[9] - 0.05) and (ys[16] > ys[13] - 0.05) and (ys[20] > ys[17] - 0.05)
        if fist_curled:
            th_x = xs[4]
            th_y = ys[4]
            idx_mcp_x = xs[5]
            mid_mcp_x = xs[9]
            if class_id in [19, 18, 13]:
                if th_x > mid_mcp_x + 0.02 and th_x < idx_mcp_x + 0.02 and th_y < 0.22:
                    return 19, max(confidence, 0.92)
                elif th_x <= mid_mcp_x - 0.02 and class_id == 19:
                    return 18, max(confidence, 0.88)

    return class_id, confidence

# Test all samples
correct = 0
for i in range(len(labels)):
    p_o = probs_orig[i]
    p_m = probs_mirr[i]
    best_p = p_m if np.max(p_m) > np.max(p_o) else p_o
    active_f = data_mirr[i] if np.max(p_m) > np.max(p_o) else data[i]
    idx = int(np.argmax(best_p))
    raw_class = classes[idx]
    conf = float(best_p[idx])
    ref_class, _ = refined_disambiguation_j_y(raw_class, conf, active_f)
    if ref_class == labels[i]:
        correct += 1
    else:
        print(f"Sample {i}: True={labels[i]}, Raw={raw_class}, Refined={ref_class}")

print(f"\nAccuracy across all 3,175 samples: {correct}/{len(labels)} ({correct/len(labels)*100:.2f}%)")

# Test specifically J (9), I (8), Y (24)
for target_lbl, target_name in [(9, 'J'), (8, 'I'), (24, 'Y'), (17, 'R'), (20, 'U'), (6, 'G'), (19, 'T')]:
    mask = (labels == target_lbl)
    total = np.sum(mask)
    cor = 0
    for i in np.where(mask)[0]:
        p_o = probs_orig[i]
        p_m = probs_mirr[i]
        best_p = p_m if np.max(p_m) > np.max(p_o) else p_o
        active_f = data_mirr[i] if np.max(p_m) > np.max(p_o) else data[i]
        idx = int(np.argmax(best_p))
        raw_class = classes[idx]
        conf = float(best_p[idx])
        ref_class, _ = refined_disambiguation_j_y(raw_class, conf, active_f)
        if ref_class == target_lbl:
            cor += 1
    print(f"Class {target_name} ({target_lbl}): {cor}/{total} ({cor/total*100:.1f}%)")
