import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1, min_detection_confidence=0.3)

for c in [6, 19]:
    detected = 0
    p = f'./data/{c}'
    imgs = os.listdir(p)
    for f in imgs:
        img = cv2.imread(os.path.join(p, f))
        if img is None: continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            detected += 1
    print(f"Class {c}: {detected}/{len(imgs)} detected with model_complexity=1, conf=0.3")
