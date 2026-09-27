import os, cv2, mediapipe as mp, pickle, numpy as np

with open('model.p', 'rb') as f:
    clf = pickle.load(f)['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0, min_detection_confidence=0.15)

for c in [6, 9, 12, 13, 17]:
    dir_p = f'data/{c}'
    files = sorted(os.listdir(dir_p), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)[:10]
    print(f"\n--- Class {c} ({dir_p}) ---")
    detected = 0
    predictions = []
    confidences = []
    for f_name in files:
        img_path = os.path.join(dir_p, f_name)
        img = cv2.imread(img_path)
        if img is None: continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            # Try flipped
            res_flip = hands.process(cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB))
            if not res_flip.multi_hand_landmarks:
                continue
            res = res_flip
            print(f"  Sample {f_name}: detected ONLY when flipped!")
        
        detected += 1
        lm = res.multi_hand_landmarks[0]
        xs = [l.x for l in lm.landmark]
        ys = [l.y for l in lm.landmark]
        mx, my = min(xs), min(ys)
        sc = max(max(xs)-mx, max(ys)-my) or 1.0
        feat = []
        for l in lm.landmark:
            feat.extend([(l.x-mx)/sc, (l.y-my)/sc])
        
        proba = clf.predict_proba([feat])[0]
        best = np.argmax(proba)
        pred_c = clf.classes_[best]
        conf = proba[best]
        predictions.append(pred_c)
        confidences.append(conf)

    print(f"Detected: {detected}/{len(files)}")
    print(f"Predictions on actual images: {predictions}")
    print(f"Confidences: {[round(c, 2) for c in confidences]}")
