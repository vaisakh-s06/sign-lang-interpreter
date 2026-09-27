import os
import cv2
import pickle
import numpy as np

# Let's inspect data/9 (J) images and labels
files_9 = os.listdir('data/9')
print(f"data/9 contains {len(files_9)} images.")

# Let's check landmark positions for class 9
with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

idxs_9 = [i for i, lbl in enumerate(d['labels']) if str(lbl) == '9']
print(f"Class 9 has {len(idxs_9)} samples in data.pickle")

for i in idxs_9[:5]:
    sample = d['data'][i]
    xs = sample[0::2]
    ys = sample[1::2]
    # Landmark 20 is pinky tip, 4 is thumb tip, 8 is index tip
    print(f"Sample {i}: thumb_tip=({xs[4]:.2f}, {ys[4]:.2f}), idx_tip=({xs[8]:.2f}, {ys[8]:.2f}), pky_tip=({xs[20]:.2f}, {ys[20]:.2f})")
