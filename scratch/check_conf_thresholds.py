import os
import cv2
import mediapipe as mp
import numpy as np

for conf in [0.35, 0.40, 0.45, 0.50, 0.55]:
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=conf
    )
    
    for c in [6, 19]:
        p = f'./data/{c}'
        imgs = os.listdir(p)[:50]
        count = 0
        for f in imgs:
            img = cv2.imread(os.path.join(p, f))
            if img is None: continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)
            if res.multi_hand_landmarks:
                count += 1
        print(f"Conf={conf:.2f} | Class {c}: {count}/50")
    detector.close()
