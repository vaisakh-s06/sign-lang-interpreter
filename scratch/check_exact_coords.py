import cv2, os, mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0)

names = {6: 'G', 9: 'J', 12: 'M', 13: 'N', 17: 'R'}

for c in [6, 9, 12, 13, 17]:
    img = cv2.imread(f'data/{c}/0.jpg')
    if img is None: continue
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        print(f"Class {c} ({names[c]}): no landmarks detected in 0.jpg")
        continue
    lm = res.multi_hand_landmarks[0].landmark
    print(f"\n--- Class {c} ({names[c]}) 0.jpg ---")
    # Print key landmark positions:
    # 0: wrist
    # 4: thumb tip
    # 8: index tip
    # 12: middle tip
    # 16: ring tip
    # 20: pinky tip
    print(f"Wrist (0) : ({lm[0].x:.3f}, {lm[0].y:.3f})")
    print(f"Thumb (4) : ({lm[4].x:.3f}, {lm[4].y:.3f}) [vs MCP 2: ({lm[2].x:.3f}, {lm[2].y:.3f})]")
    print(f"Index (8) : ({lm[8].x:.3f}, {lm[8].y:.3f}) [vs MCP 5: ({lm[5].x:.3f}, {lm[5].y:.3f})]")
    print(f"Middle(12): ({lm[12].x:.3f}, {lm[12].y:.3f}) [vs MCP 9: ({lm[9].x:.3f}, {lm[9].y:.3f})]")
    print(f"Ring  (16): ({lm[16].x:.3f}, {lm[16].y:.3f}) [vs MCP 13: ({lm[13].x:.3f}, {lm[13].y:.3f})]")
    print(f"Pinky (20): ({lm[20].x:.3f}, {lm[20].y:.3f}) [vs MCP 17: ({lm[17].x:.3f}, {lm[17].y:.3f})]")
