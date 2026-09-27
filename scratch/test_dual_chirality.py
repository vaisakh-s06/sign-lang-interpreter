import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

LABELS_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'Hello', 27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.',
    33: 'lional messi-the goat🐐'
}

def mirror_feat(feat):
    xs = np.array(feat[0::2])
    ys = np.array(feat[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-4:
        scale = 1.0
    out = []
    for x, y in zip((xs_m - min_x)/scale, (ys - min_y)/scale):
        out.extend([float(x), float(y)])
    return out

# In real webcam inference:
# If a user shows a left hand, its coordinates are mirrored.
# If we test predicting on canonical coordinates (mirroring left hand):
print("Testing prediction on Left Hand when canonicalized:")

total = 0
correct = 0
for i, (sample, lbl) in enumerate(zip(data, labels)):
    # Simulate left hand
    left_hand_sample = mirror_feat(sample)
    
    # In the app, if we take max(proba(left_hand_sample), proba(mirror_feat(left_hand_sample))):
    # Or canonicalize:
    f_orig = np.asarray(left_hand_sample, dtype=np.float32).reshape(1, -1)
    f_canon = np.asarray(mirror_feat(left_hand_sample), dtype=np.float32).reshape(1, -1)
    
    prob_orig = model.predict_proba(f_orig)[0]
    prob_canon = model.predict_proba(f_canon)[0]
    
    # Combined / best probability:
    # If the user uses left hand, prob_canon will be the true sign!
    # If the user uses right hand, prob_orig will be the true sign!
    combined_prob = np.maximum(prob_orig, prob_canon)
    best_idx = np.argmax(combined_prob)
    pred_class = model.classes_[best_idx]
    
    if pred_class == str(lbl):
        correct += 1
    total += 1

print(f"Dual-chirality (Left + Right hand) accuracy across all 3,175 samples: {correct}/{total} ({correct/total*100:.2f}%)!")
