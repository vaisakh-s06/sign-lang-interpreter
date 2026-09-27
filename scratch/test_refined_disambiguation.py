import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

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

# Batch predict
print("Predicting batch...")
probs_orig = model.predict_proba(data)
probs_mirr = model.predict_proba(data_mirr)
print("Batch prediction done.")

def refined_disambiguation(class_id, confidence, data_aux):
    if len(data_aux) != 42:
        return class_id, confidence

    xs = data_aux[0::2]
    ys = data_aux[1::2]

    # Finger extension states
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
            return 21 if tip_dist > 0.15 else 20, max(confidence, 0.88)

    # --- 2. D (3) vs Z (25) ONLY (only when middle finger is NOT up) ---
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
        if cross_metric < -0.0001 or abs(xs[8] - xs[12]) < 0.04:
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

    return class_id, confidence

classes = np.array([int(c) for c in model.classes_])
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

print(f"Accuracy with refined disambiguation: {correct}/{len(labels)} ({correct/len(labels)*100:.2f}%)")
