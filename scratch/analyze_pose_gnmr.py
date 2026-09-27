import os, cv2, mediapipe as mp, numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0)

def describe_hand(lm):
    # Check finger states: extended or curled
    # Index: 8 vs 6
    idx_ext = lm[8].y < lm[6].y
    # Middle: 12 vs 10
    mid_ext = lm[12].y < lm[10].y
    # Ring: 16 vs 14
    ring_ext = lm[16].y < lm[14].y
    # Pinky: 20 vs 18
    pky_ext = lm[20].y < lm[18].y
    # Thumb: 4 vs 2
    th_dx = lm[4].x - lm[2].x
    th_dy = lm[4].y - lm[2].y

    # Horizontal span of index
    idx_dx = lm[8].x - lm[5].x
    idx_dy = lm[8].y - lm[5].y

    return {
        'idx_ext_up': idx_ext,
        'mid_ext_up': mid_ext,
        'ring_ext_up': ring_ext,
        'pky_ext_up': pky_ext,
        'idx_dir': (round(idx_dx, 2), round(idx_dy, 2)),
        'th_dir': (round(th_dx, 2), round(th_dy, 2)),
    }

for c in [6, 9, 12, 13, 17]:
    names = {6: 'G', 9: 'J', 12: 'M', 13: 'N', 17: 'R'}
    img_p = f'data/{c}/0.jpg'
    img = cv2.imread(img_p)
    if img is not None:
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            desc = describe_hand(res.multi_hand_landmarks[0].landmark)
            print(f"Class {c:2d} ({names[c]}): {desc}")
        else:
            print(f"Class {c:2d} ({names[c]}): landmark not detected in 0.jpg")
