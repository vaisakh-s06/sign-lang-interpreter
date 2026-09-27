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

def targeted_disambiguation(class_id, confidence, data_aux):
    """
    Carefully resolves ONLY the 3 specific confusions requested by the user,
    leaving ALL other alphabets and words 100% untouched:
      1. D (3) vs Z (25): D points straight up (lean >= -0.20), Z leans to the left (lean < -0.22).
      2. R (17) vs U (20)/V (21): R has crossed index and middle fingers.
      3. J (9) vs I (8): J has hooked/tilted pinky or elevated thumb, I is straight vertical pinky.
    """
    if len(data_aux) != 42:
        return class_id, confidence

    xs = data_aux[0::2]
    ys = data_aux[1::2]

    # --- 1. D (3) vs Z (25) ONLY ---
    if class_id in [3, 25]:
        dx = xs[8] - xs[5]
        dy = ys[5] - ys[8]  # dy > 0 when index points up
        if dy > 0.05:
            lean = dx / dy
            # D points straight up (mean -0.05, range -0.10 to 0.00)
            # Z leans distinctly to the left (mean -0.67, range -0.75 to -0.56)
            if lean < -0.22:
                return 25, max(confidence, 0.90)  # Z: leaning to left
            elif lean >= -0.20:
                return 3, max(confidence, 0.90)   # D: pointing straight up

    # --- 2. R (17) vs U (20)/V (21) ONLY ---
    elif class_id in [17, 20, 21]:
        knuckle_dx = xs[5] - xs[9]
        tip_dx = xs[8] - xs[12]
        cross_metric = knuckle_dx * tip_dx
        # If index and middle fingers are crossed:
        if cross_metric < -0.0002:
            return 17, max(confidence, 0.92)  # R: crossed fingers
        elif cross_metric > 0.001 and class_id == 17:
            # Fingers are clearly uncrossed and parallel: U or V
            u_v_dist = abs(xs[8] - xs[12])
            new_id = 21 if u_v_dist > 0.15 else 20
            return new_id, max(confidence, 0.85)

    # --- 3. J (9) vs I (8) ONLY ---
    elif class_id in [8, 9]:
        pinky_dx = xs[20] - xs[17]
        thumb_higher = ys[4] < ys[20] + 0.05
        # In J: pinky is hooked to left (pinky_dx < -0.18) or thumb is higher than pinky tip
        if pinky_dx < -0.18 or thumb_higher:
            return 9, max(confidence, 0.90)   # J: hooked/tilted pinky or thumb up
        else:
            return 8, max(confidence, 0.90)   # I: straight upright pinky

    # ALL OTHER ALPHABETS AND WORDS: 100% UNTOUCHED
    return class_id, confidence

# Let's test on all 3,175 samples
X_all = np.asarray(data, dtype=np.float32)
all_probas = model.predict_proba(X_all)
best_indices = np.argmax(all_probas, axis=1)

total = len(data)
correct = 0
mismatches = []

unique_labels = sorted(list(set(labels)), key=lambda x: int(x) if x.isdigit() else 999)

for lbl in unique_labels:
    indices = [i for i, l in enumerate(labels) if str(l) == str(lbl)]
    c_correct = 0
    for idx in indices:
        feat = data[idx]
        proba = all_probas[idx]
        orig_id = int(str(model.classes_[best_indices[idx]]))
        orig_conf = float(proba[best_indices[idx]])
        
        final_id, final_conf = targeted_disambiguation(orig_id, orig_conf, feat)
        if final_id == int(lbl):
            c_correct += 1
            correct += 1
        else:
            mismatches.append((lbl, final_id, LABELS_DICT.get(int(lbl)), LABELS_DICT.get(final_id)))
    
    char_name = LABELS_DICT.get(int(lbl), f"Sign {lbl}")
    print(f"Class {lbl:>2} ({char_name:<16}): {c_correct}/{len(indices)} ({c_correct/len(indices)*100:.1f}%)")

print(f"\nOVERALL ACCURACY ACROSS ALL SAMPLES: {correct}/{total} ({correct/total*100:.2f}%)")
if mismatches:
    print(f"Mismatches ({len(mismatches)}):", mismatches)
else:
    print("PERFECT! 100% ACCURACY ACROSS ALL 3,175 SAMPLES!")
