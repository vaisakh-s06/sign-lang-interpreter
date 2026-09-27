import os
import cv2
import mediapipe as mp
import numpy as np

for c in [6, 19]:
    p = f'./data/{c}'
    imgs = os.listdir(p)
    
    for complexity in [0, 1]:
        mp_hands = mp.solutions.hands
        detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=complexity,
            min_detection_confidence=0.20
        )
        
        scores = []
        areas = []
        count = 0
        for f in imgs[:50]:
            img = cv2.imread(os.path.join(p, f))
            if img is None: continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)
            if res.multi_hand_landmarks:
                count += 1
                if res.multi_handedness:
                    scores.append(res.multi_handedness[0].classification[0].score)
                hl = res.multi_hand_landmarks[0]
                hx = [lm.x for lm in hl.landmark]
                hy = [lm.y for lm in hl.landmark]
                areas.append((max(hx) - min(hx)) * (max(hy) - min(hy)))
        
        detector.close()
        avg_score = np.mean(scores) if scores else 0
        min_score = np.min(scores) if scores else 0
        print(f"Class {c} | complexity={complexity}: Detected {count}/50 | Avg score: {avg_score:.3f} | Min score: {min_score:.3f}")
