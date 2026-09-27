def is_pointing_up(landmarks):
    lm = landmarks.landmark
    # 0: wrist
    # 5,6,7,8: index mcp, pip, dip, tip
    # 9,10,11,12: middle
    # 13,14,15,16: ring
    # 17,18,19,20: pinky
    
    idx_tip = lm[8]
    idx_pip = lm[6]
    idx_mcp = lm[5]
    wrist = lm[0]

    # 1. Index finger extended upwards (tip above pip, pip above mcp, and tip above wrist)
    idx_extended = (idx_tip.y < idx_pip.y) and (idx_tip.y < idx_mcp.y) and (idx_tip.y < wrist.y - 0.05)
    
    # 2. Angle: mostly upward (vertical component dy dominates or is reasonably large)
    dy = abs(idx_tip.y - idx_mcp.y)
    dx = abs(idx_tip.x - idx_mcp.x)
    mostly_upward = dy > (dx * 0.5)

    # 3. Middle, ring, and pinky are curled down (tips significantly lower than index tip)
    mid_curled = (lm[12].y > idx_tip.y + 0.04) or (lm[12].y > lm[10].y)
    ring_curled = (lm[16].y > idx_tip.y + 0.04) or (lm[16].y > lm[14].y)
    pinky_curled = (lm[20].y > idx_tip.y + 0.04) or (lm[20].y > lm[18].y)

    return idx_extended and mostly_upward and mid_curled and ring_curled and pinky_curled

def detect_messi_celebration(multi_hand_landmarks):
    if not multi_hand_landmarks or len(multi_hand_landmarks) < 2:
        return False
    # Check if both hands are pointing up
    hand1_up = is_pointing_up(multi_hand_landmarks[0])
    hand2_up = is_pointing_up(multi_hand_landmarks[1])
    return hand1_up and hand2_up

print("Detection helper ready")
