import os, cv2, mediapipe as mp, pickle, numpy as np

with open('model.p', 'rb') as f:
    clf = pickle.load(f)['model']

# Check classes 6 (G), 9 (J), 12 (M), 13 (N), 17 (R) in data.pickle
with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(str(x)) for x in d['labels']])

names = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.'
}

# Let's inspect the actual image files in data/6, data/9, data/12, data/13, data/17
# to see what poses are in those images.
for c in [6, 9, 12, 13, 17]:
    dir_p = f'data/{c}'
    files = [f for f in os.listdir(dir_p) if f.endswith('.jpg')]
    print(f"\n=================== Class {c}: {names[c]} ({len(files)} images) ===================")
    
    # Check sample 0, 25, 50, 75
    for idx in [0, 25, 50, 75]:
        img_p = os.path.join(dir_p, f"{idx}.jpg")
        if not os.path.exists(img_p): continue
        img = cv2.imread(img_p)
        if img is None: continue
        h, w, _ = img.shape
        print(f"  Image {idx}.jpg size: {w}x{h}")
