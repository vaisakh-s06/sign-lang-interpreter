import cv2, mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0)
mp_drawing = mp.solutions.drawing_utils

for c in [6, 9, 12, 13, 17]:
    img = cv2.imread(f'data/{c}/0.jpg')
    if img is not None:
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.imwrite(f'scratch/sample_{c}.jpg', img)

print("Saved sample images with landmarks to scratch/")
