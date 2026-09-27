import os
import cv2
import mediapipe as mp
import numpy as np

# Load one image from data/6 and data/19
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5)

for c, name in [(6, 'G'), (19, 'T')]:
    img_path = f'./data/{c}/0.jpg'
    img = cv2.imread(img_path)
    if img is not None:
        H, W, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            print(f"\n--- {name} (Class {c}) Image 0.jpg (size {W}x{H}) ---")
            print(f"Wrist (0): x={lm.landmark[0].x:.3f}, y={lm.landmark[0].y:.3f}")
            print(f"Thumb tip (4): x={lm.landmark[4].x:.3f}, y={lm.landmark[4].y:.3f}")
            print(f"Index tip (8): x={lm.landmark[8].x:.3f}, y={lm.landmark[8].y:.3f}")
            print(f"Middle tip (12): x={lm.landmark[12].x:.3f}, y={lm.landmark[12].y:.3f}")
            print(f"Ring tip (16): x={lm.landmark[16].x:.3f}, y={lm.landmark[16].y:.3f}")
            print(f"Pinky tip (20): x={lm.landmark[20].x:.3f}, y={lm.landmark[20].y:.3f}")
            
            # Check orientation:
            # dx = index_tip - index_mcp
            # dy = index_tip - index_mcp
            idx_mcp = lm.landmark[5]
            idx_tip = lm.landmark[8]
            print(f"Index vector: dx={idx_tip.x - idx_mcp.x:.3f}, dy={idx_tip.y - idx_mcp.y:.3f}")
