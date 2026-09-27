import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

img = cv2.imread('data/0/0.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res = hands.process(rgb)

if res.multi_handedness:
    for h in res.multi_handedness:
        print("Handedness label for dataset image:", h.classification[0].label, "score:", h.classification[0].score)
        
hands.close()
