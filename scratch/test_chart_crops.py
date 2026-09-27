import cv2, mediapipe as mp, pickle, numpy as np

with open('model.p', 'rb') as f:
    clf = pickle.load(f)['model']

chart = cv2.imread('scratch/chart.png')
# Chart is 750 wide, 393 high
# Row 1: A, B, C, D, E, F, G, H, I (y: 0 to 125)
# Row 2: J, K, L, M, N, O, P, Q (y: 125 to 260)
# Row 3: R, S, T, U, V, W, X, Y, Z (y: 260 to 393)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0, min_detection_confidence=0.1)

names = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.'
}

# Approximate crops for G, J, M, N, R
# G is top row, around x: 420 to 530, y: 30 to 125
# J is row 2, around x: 15 to 110, y: 140 to 250
# M is row 2, around x: 260 to 350, y: 150 to 250
# N is row 2, around x: 350 to 430, y: 150 to 250
# R is row 3, around x: 15 to 80, y: 260 to 380

crops = {
    'G': chart[20:130, 420:530],
    'J': chart[135:255, 10:120],
    'M': chart[140:255, 260:350],
    'N': chart[140:255, 345:430],
    'R': chart[260:390, 15:85]
}

for label, crp in crops.items():
    res = hands.process(cv2.cvtColor(crp, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        print(f"Letter {label}: MediaPipe detected NO landmarks on cropped chart illustration.")
    else:
        lm = res.multi_hand_landmarks[0]
        xs = [l.x for l in lm.landmark]
        ys = [l.y for l in lm.landmark]
        mx, my = min(xs), min(ys)
        sc = max(max(xs)-mx, max(ys)-my) or 1.0
        feat = []
        for l in lm.landmark:
            feat.extend([(l.x-mx)/sc, (l.y-my)/sc])
        proba = clf.predict_proba([feat])[0]
        best = np.argmax(proba)
        pred = names[clf.classes_[best]]
        top3 = sorted([(names[clf.classes_[i]], proba[i]) for i in range(len(proba))], key=lambda x: x[1], reverse=True)[:3]
        print(f"Letter {label}: Predicted = {pred} ({top3[0][1]*100:.1f}%) | Top 3: {top3}")
