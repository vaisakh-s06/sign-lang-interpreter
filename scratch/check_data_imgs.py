import os
import cv2
import pickle
import numpy as np

# Check how many images exist in data/6 and data/19
for c in [6, 19]:
    p = f'./data/{c}'
    if os.path.exists(p):
        imgs = [f for f in os.listdir(p) if f.endswith('.jpg')]
        print(f"data/{c} has {len(imgs)} images.")

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

print(f"Samples in data.pickle for G (6): {np.sum(labels == 6)}")
print(f"Samples in data.pickle for T (19): {np.sum(labels == 19)}")
