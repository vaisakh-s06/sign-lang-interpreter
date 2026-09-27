import pickle, numpy as np

with open('model.p', 'rb') as f:
    clf = pickle.load(f)['model']
with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(str(x)) for x in d['labels']])

names = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.'
}

def refine_prediction(pred_id, proba, feat_42):
    # feat_42: list of 42 normalized coords
    # 5: index mcp (x: 10, y: 11), 8: index tip (x: 16, y: 17)
    # 9: mid mcp   (x: 18, y: 19), 12: mid tip  (x: 24, y: 25)
    # 13: ring mcp (x: 26, y: 27), 16: ring tip (x: 32, y: 33)
    # 17: pky mcp  (x: 34, y: 35), 20: pky tip  (x: 40, y: 41)
    # 2: thumb mcp (x: 4, y: 5),   4: thumb tip (x: 8, y: 9)
    
    # Check 1: R (17) vs U (20) / V (21)
    # When model predicts R or U or V:
    if pred_id in [17, 20, 21]:
        knuckle_dx = feat_42[10] - feat_42[18]
        tip_dx = feat_42[16] - feat_42[24]
        is_crossed = (knuckle_dx * tip_dx < -0.0005)
        if is_crossed:
            return 17, max(proba[pred_id], 0.85) # Firmly R!
        elif pred_id == 17:
            # Not crossed but predicted R -> should be U!
            return 20, max(proba[pred_id], 0.85)

    # Check 2: M (12) vs N (13) vs fist family
    if pred_id in [12, 13]:
        ring_y = feat_42[33]
        mid_y = feat_42[25]
        # In M, ring is level with mid (diff < 0.15)
        # In N, ring is curled down (diff >= 0.15)
        if (ring_y - mid_y) < 0.15:
            return 12, max(proba[pred_id], 0.85) # M!
        else:
            return 13, max(proba[pred_id], 0.85) # N!

    # Check 3: G (6) vs H (7)
    if pred_id in [6, 7]:
        idx_tip_x = feat_42[16]
        mid_tip_x = feat_42[24]
        idx_tip_y = feat_42[17]
        mid_tip_y = feat_42[25]
        dist = np.sqrt((idx_tip_x - mid_tip_x)**2 + (idx_tip_y - mid_tip_y)**2)
        if dist > 0.45:
            return 6, max(proba[pred_id], 0.85) # G!
        else:
            return 7, max(proba[pred_id], 0.85) # H!

    # Check 4: J (9) vs I (8)
    if pred_id in [8, 9]:
        th_y = feat_42[9]
        pky_y = feat_42[41]
        pky_dx = abs(feat_42[40] - feat_42[34])
        # In J, thumb is higher than pinky tip or pinky is hooked sideways
        if th_y < pky_y or pky_dx > 0.20:
            return 9, max(proba[pred_id], 0.85) # J!
        else:
            return 8, max(proba[pred_id], 0.85) # I!

    return pred_id, proba[pred_id]

# Test on all samples in data.pickle
correct = 0
total = len(data)
for feat, lbl in zip(data, labels):
    p = clf.predict_proba([feat])[0]
    raw_pred = int(str(clf.classes_[np.argmax(p)]))
    ref_pred, conf = refine_prediction(raw_pred, {int(str(clf.classes_[i])): p[i] for i in range(len(p))}, feat)
    if ref_pred == lbl:
        correct += 1

print(f"Overall Accuracy with stabilizer: {correct}/{total} = {correct/total*100:.2f}%")

# Test specifically G, J, M, N, R
for c in [6, 9, 12, 13, 17]:
    idx = (labels == c)
    c_samples = data[idx]
    c_correct = 0
    for s in c_samples:
        p = clf.predict_proba([s])[0]
        raw_p = int(str(clf.classes_[np.argmax(p)]))
        ref_p, conf = refine_prediction(raw_p, {int(str(clf.classes_[i])): p[i] for i in range(len(p))}, s)
        if ref_p == c: c_correct += 1
    print(f"Class {c:2d} ({names[c]}): {c_correct}/{len(c_samples)} ({c_correct/len(c_samples)*100:.1f}%)")
