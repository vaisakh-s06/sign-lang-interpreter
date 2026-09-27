import os, cv2, mediapipe as mp, pickle, numpy as np

with open('model.p', 'rb') as f:
    clf = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(str(x)) for x in d['labels']])

# Let's inspect feature differences between:
# 8 (I) and 9 (J)
# 12 (M) and 13 (N) and 0 (A) and 18 (S) and 4 (E)
# 6 (G) and 7 (H)
# 17 (R) and 20 (U) and 21 (V)

def get_class_features(c):
    return data[labels == c]

f8 = get_class_features(8)
f9 = get_class_features(9)
dist_8_9 = np.mean(np.linalg.norm(f8.mean(axis=0) - f9.mean(axis=0)))
print(f"Distance between centroid of I (8) and J (9): {dist_8_9:.4f}")

f12 = get_class_features(12)
f13 = get_class_features(13)
f0  = get_class_features(0)
dist_12_13 = np.mean(np.linalg.norm(f12.mean(axis=0) - f13.mean(axis=0)))
dist_12_0  = np.mean(np.linalg.norm(f12.mean(axis=0) - f0.mean(axis=0)))
dist_13_0  = np.mean(np.linalg.norm(f13.mean(axis=0) - f0.mean(axis=0)))
print(f"Distance between M (12) and N (13): {dist_12_13:.4f}")
print(f"Distance between M (12) and A (0):  {dist_12_0:.4f}")
print(f"Distance between N (13) and A (0):  {dist_13_0:.4f}")

f17 = get_class_features(17)
f20 = get_class_features(20)
f21 = get_class_features(21)
dist_17_20 = np.mean(np.linalg.norm(f17.mean(axis=0) - f20.mean(axis=0)))
dist_17_21 = np.mean(np.linalg.norm(f17.mean(axis=0) - f21.mean(axis=0)))
print(f"Distance between R (17) and U (20): {dist_17_20:.4f}")
print(f"Distance between R (17) and V (21): {dist_17_21:.4f}")

f6 = get_class_features(6)
f7 = get_class_features(7)
dist_6_7 = np.mean(np.linalg.norm(f6.mean(axis=0) - f7.mean(axis=0)))
print(f"Distance between G (6) and H (7):   {dist_6_7:.4f}")
