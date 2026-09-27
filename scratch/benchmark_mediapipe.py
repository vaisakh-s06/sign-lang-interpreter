import mediapipe as mp
import cv2
import time
import numpy as np

# Test model_complexity=1 vs 0
mp_hands = mp.solutions.hands

dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
# Add some wall texture / shadows
cv2.rectangle(dummy_frame, (100, 100), (300, 300), (120, 120, 120), -1)
cv2.line(dummy_frame, (0, 200), (640, 250), (80, 80, 80), 3)

for mc in [0, 1]:
    h = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=mc,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    t0 = time.time()
    for _ in range(15):
        res = h.process(dummy_frame)
    t1 = time.time()
    dt = (t1 - t0) / 15 * 1000
    detected = len(res.multi_hand_landmarks) if res.multi_hand_landmarks else 0
    print(f"model_complexity={mc}: avg latency = {dt:.1f}ms, false detection on wall texture = {detected}")
    h.close()
