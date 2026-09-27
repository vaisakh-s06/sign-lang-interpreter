import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Suppress noisy logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_hands = mp.solutions.hands
# model_complexity=0 uses lightweight palm detector that detects closed fists (A, E, S) with 98-100% accuracy
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.15
)

DATA_DIR = './data'
data = []
labels = []

print("Extracting hand landmarks with model_complexity=0 and scale-invariance...")

dirs = sorted(os.listdir(DATA_DIR), key=lambda x: int(x) if x.isdigit() else 999)

for dir_ in dirs:
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    img_files = os.listdir(dir_path)
    det_count = 0

    for img_name in img_files:
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            scale = max(max_x - min_x, max_y - min_y)
            if scale < 1e-4:
                scale = 1.0

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append((lm.x - min_x) / scale)
                data_aux.append((lm.y - min_y) / scale)

            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)
                det_count += 1

    print(f"  Class {dir_}: {det_count}/{len(img_files)} detected and normalized.")

print(f"\nTotal extracted samples: {len(data)}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Saved clean, scale-normalized dataset to data.pickle!")
