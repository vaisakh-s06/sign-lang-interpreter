import os, cv2, mediapipe as mp, pickle, numpy as np

with open('model.p', 'rb') as f:
    clf = pickle.load(f)['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0, min_detection_confidence=0.15)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.'
}

def get_feat(lm):
    xs = [l.x for l in lm.landmark]
    ys = [l.y for l in lm.landmark]
    mx, my = min(xs), min(ys)
    sc = max(max(xs)-mx, max(ys)-my) or 1.0
    feat = []
    for l in lm.landmark:
        feat.extend([(l.x-mx)/sc, (l.y-my)/sc])
    return feat

for c in [6, 8, 9, 12, 13, 17, 20, 21]:
    dir_p = f'data/{c}'
    files = sorted(os.listdir(dir_p))[:30]
    
    unflipped_preds = {}
    flipped_preds = {}
    
    for f_name in files:
        img = cv2.imread(os.path.join(dir_p, f_name))
        if img is None: continue
        
        # 1. Unflipped
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            feat = get_feat(res.multi_hand_landmarks[0])
            pred = int(str(clf.predict([feat])[0]))
            pred_name = labels_dict.get(pred, str(pred))
            unflipped_preds[pred_name] = unflipped_preds.get(pred_name, 0) + 1

        # 2. Flipped (horizontal flip, exactly what app.py does!)
        res_f = hands.process(cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB))
        if res_f.multi_hand_landmarks:
            feat_f = get_feat(res_f.multi_hand_landmarks[0])
            pred_f = int(str(clf.predict([feat_f])[0]))
            pred_name_f = labels_dict.get(pred_f, str(pred_f))
            flipped_preds[pred_name_f] = flipped_preds.get(pred_name_f, 0) + 1

    print(f"Class {c:2d} ({labels_dict[c]}):")
    print(f"   Unflipped: {unflipped_preds}")
    print(f"   Flipped  : {flipped_preds}")
