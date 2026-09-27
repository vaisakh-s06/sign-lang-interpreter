import numpy as np

def check_hand_pointing_up(landmarks):
    # landmarks: list of 21 (x, y, z) objects
    lm = landmarks
    # Index finger extended upward:
    # y is 0 at top, 1 at bottom. So smaller y is higher up.
    idx_tip = lm[8]
    idx_dip = lm[7]
    idx_pip = lm[6]
    idx_mcp = lm[5]
    wrist = lm[0]

    # Index finger pointing up: tip is higher than pip, mcp, and wrist
    idx_up = (idx_tip.y < idx_pip.y) and (idx_tip.y < idx_mcp.y) and (idx_tip.y < wrist.y - 0.08)
    
    # Check vertical orientation (not pointing sideways)
    dy = abs(idx_tip.y - idx_mcp.y)
    dx = abs(idx_tip.x - idx_mcp.x)
    mostly_vertical = (dy > dx * 0.7)

    # Middle, ring, pinky fingers are curled or distinctly lower than index tip
    mid_curled = (lm[12].y > idx_tip.y + 0.05)
    ring_curled = (lm[16].y > idx_tip.y + 0.05)
    pinky_curled = (lm[20].y > idx_tip.y + 0.05)

    return idx_up and mostly_vertical and mid_curled and ring_curled and pinky_curled

class DummyLM:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Create mock landmarks for a hand pointing up
lms = [DummyLM(0.5, 0.8)] # wrist 0
for i in range(1, 5): lms.append(DummyLM(0.4, 0.7)) # thumb 1-4
lms.append(DummyLM(0.5, 0.6)) # idx mcp 5
lms.append(DummyLM(0.5, 0.45)) # idx pip 6
lms.append(DummyLM(0.5, 0.35)) # idx dip 7
lms.append(DummyLM(0.5, 0.25)) # idx tip 8
for i in range(9, 13): lms.append(DummyLM(0.55, 0.65)) # mid 9-12
for i in range(13, 17): lms.append(DummyLM(0.6, 0.65)) # ring 13-16
for i in range(17, 21): lms.append(DummyLM(0.65, 0.65)) # pinky 17-20

print("Mock hand pointing up result:", check_hand_pointing_up(lms))
