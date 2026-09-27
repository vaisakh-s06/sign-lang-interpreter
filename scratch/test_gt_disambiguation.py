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

def refined_disambiguation(class_id, confidence, data_aux):
    if len(data_aux) != 42:
        return class_id, confidence

    xs = data_aux[0::2]
    ys = data_aux[1::2]

    # Finger extension states (smaller y means higher in the image frame)
    idx_up = ys[8] < ys[6]
    mid_up = ys[12] < ys[10]

    # --- 1. Prevent D when two fingers are pointing upwards (R vs D) ---
    if class_id == 3 and mid_up:
        knuckle_dx = xs[5] - xs[9]
        tip_dx = xs[8] - xs[12]
        cross_metric = knuckle_dx * tip_dx
        tip_dist = abs(xs[8] - xs[12])
        if cross_metric < 0.001 or tip_dist < 0.06:
            return 17, max(confidence, 0.92)  # R: crossed/twisted fingers
        else:
            return (21 if tip_dist > 0.15 else 20), max(confidence, 0.88)

    # --- 2. D (3) vs Z (25) ONLY (only valid when middle finger is NOT up) ---
    if class_id in [3, 25]:
        if mid_up:
            return 17, max(confidence, 0.90)
        dx = xs[8] - xs[5]
        dy = ys[5] - ys[8]
        if dy > 0.05:
            lean = dx / dy
            if lean < -0.22:
                return 25, max(confidence, 0.90)  # Z: leaning to left
            elif lean >= -0.20:
                return 3, max(confidence, 0.90)   # D: pointing straight up

    # --- 3. R (17) vs U (20)/V (21) ONLY ---
    elif class_id in [17, 20, 21]:
        knuckle_dx = xs[5] - xs[9]
        tip_dx = xs[8] - xs[12]
        cross_metric = knuckle_dx * tip_dx
        if cross_metric < -0.0001 or abs(xs[8] - xs[12]) < 0.05:
            return 17, max(confidence, 0.92)  # R: crossed fingers
        elif cross_metric > 0.001 and class_id == 17:
            u_v_dist = abs(xs[8] - xs[12])
            new_id = 21 if u_v_dist > 0.15 else 20
            return new_id, max(confidence, 0.85)

    # --- 4. J (9) vs I (8) ONLY ---
    elif class_id in [8, 9]:
        pinky_dx = xs[20] - xs[17]
        thumb_higher = ys[4] < ys[20] + 0.05
        if pinky_dx < -0.18 or thumb_higher:
            return 9, max(confidence, 0.90)   # J: hooked/tilted pinky or thumb up
        else:
            return 8, max(confidence, 0.90)   # I: straight upright pinky

    # --- 5. G (6) vs H (7) vs other index gestures ---
    # In G: index is pointing horizontally, middle is curled.
    # In H: index AND middle are pointing horizontally side-by-side.
    if class_id in [6, 7]:
        idx_horiz_ext = abs(xs[8] - xs[5])
        mid_horiz_ext = abs(xs[12] - xs[9])
        # If middle finger is clearly extended horizontally alongside index: H (7)
        if mid_horiz_ext >= 0.40 and abs(ys[8] - ys[12]) < 0.15:
            return 7, max(confidence, 0.92)  # H: two horizontal fingers
        elif idx_horiz_ext >= 0.35 and mid_horiz_ext < 0.35:
            return 6, max(confidence, 0.92)  # G: one horizontal finger

    # --- 6. T (19) vs Fist family (A=0, S=18, N=13, M=12) ---
    # When a fist is formed (fingers curled down), thumb placement distinguishes them:
    # In T: thumb tip is between index MCP (xs[5]) and middle MCP (xs[9]).
    # In A: thumb tip is on outer side (xs[4] > xs[5] + 0.06).
    if class_id in [19, 18, 13, 12, 0]:
        # Check if all 4 fingers are curled into palm
        fist_curled = (ys[8] > ys[5] - 0.05) and (ys[12] > ys[9] - 0.05) and (ys[16] > ys[13] - 0.05) and (ys[20] > ys[17] - 0.05)
        if fist_curled:
            th_x = xs[4]
            th_y = ys[4]
            idx_mcp_x = xs[5]
            mid_mcp_x = xs[9]
            # T has thumb tucked between index and middle knuckles:
            # mid_mcp_x ~ 0.30, idx_mcp_x ~ 0.46. th_x is around 0.34 to 0.44.
            if class_id in [19, 18, 13]:
                if th_x > mid_mcp_x + 0.02 and th_x < idx_mcp_x + 0.02 and th_y < 0.22:
                    return 19, max(confidence, 0.92)  # T: thumb between index and middle
                elif th_x <= mid_mcp_x - 0.02 and class_id == 19:
                    return 18, max(confidence, 0.88)  # S: thumb over middle/ring

    return class_id, confidence

# Test accuracy across all 3,175 samples
correct = 0
for i in range(len(labels)):
    p_o = probs_orig[i]
    p_m = probs_mirr[i]
    
    if np.max(p_m) > np.max(p_o):
        best_p = p_m
        active_f = data_mirr[i]
    else:
        best_p = p_o
        active_f = data[i]
        
    idx = int(np.argmax(best_p))
    raw_class = classes[idx]
    conf = float(best_p[idx])
    
    ref_class, ref_conf = refined_disambiguation(raw_class, conf, active_f)
    if ref_class == labels[i]:
        correct += 1
    else:
        print(f"Sample {i}: True={labels[i]}, Raw={raw_class}, Refined={ref_class}")

print(f"\nAccuracy across all samples: {correct}/{len(labels)} ({correct/len(labels)*100:.2f}%)")

# Check G and T specifically:
for target_lbl, target_name in [(6, 'G'), (19, 'T'), (7, 'H'), (0, 'A'), (18, 'S'), (13, 'N'), (12, 'M'), (3, 'D'), (17, 'R'), (25, 'Z')]:
    mask = (labels == target_lbl)
    true_count = np.sum(mask)
    cor_count = 0
    for i in np.where(mask)[0]:
        p_o = probs_orig[i]
        p_m = probs_mirr[i]
        best_p = p_m if np.max(p_m) > np.max(p_o) else p_o
        active_f = data_mirr[i] if np.max(p_m) > np.max(p_o) else data[i]
        idx = int(np.argmax(best_p))
        raw_class = classes[idx]
        conf = float(best_p[idx])
        ref_class, _ = refined_disambiguation(raw_class, conf, active_f)
        if ref_class == target_lbl:
            cor_count += 1
    print(f"Class {target_name} ({target_lbl}): {cor_count}/{true_count} ({cor_count/true_count*100:.1f}%)")
