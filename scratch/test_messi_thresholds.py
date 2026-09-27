import math

def is_pointing_up(lm):
    # lm: list of 21 landmark objects with .x, .y
    # In MediaPipe, y is normalized [0, 1] from top to bottom.
    idx_tip = lm[8]
    idx_dip = lm[7]
    idx_pip = lm[6]
    idx_mcp = lm[5]
    wrist = lm[0]

    # 1. Index finger extended upward:
    # Tip must be above PIP, MCP, and wrist
    if not (idx_tip.y < idx_pip.y and idx_tip.y < idx_mcp.y and idx_tip.y < wrist.y - 0.04):
        return False

    # 2. Orientation: index finger pointing generally upward (within ~65 degrees from vertical)
    dy = abs(idx_mcp.y - idx_tip.y)
    dx = abs(idx_mcp.x - idx_tip.x)
    if dy < dx * 0.45: # allows spreading arms like Messi
        return False

    # 3. Middle, Ring, Pinky curled down:
    # Their tips should be distinctly lower than the index tip
    # or below their respective PIP joints
    mid_curled = (lm[12].y > idx_tip.y + 0.035) or (lm[12].y > lm[10].y)
    ring_curled = (lm[16].y > idx_tip.y + 0.035) or (lm[16].y > lm[14].y)
    pinky_curled = (lm[20].y > idx_tip.y + 0.035) or (lm[20].y > lm[18].y)

    if not (mid_curled and ring_curled and pinky_curled):
        return False

    # 4. Index tip is the highest point of the hand
    if idx_tip.y > lm[12].y or idx_tip.y > lm[16].y or idx_tip.y > lm[20].y:
        return False

    return True

class LM:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Test angled hand (left arm raised like Messi, index angled 30 deg outward)
# Wrist at (0.3, 0.8), MCP at (0.28, 0.55), Tip at (0.22, 0.3)
# dx = 0.06, dy = 0.25 -> dy/dx = 4.16 > 0.45 -> True!
hand_left = [LM(0.3, 0.8)] # wrist 0
for _ in range(4): hand_left.append(LM(0.32, 0.6)) # thumb 1-4
hand_left.append(LM(0.28, 0.55)) # idx mcp 5
hand_left.append(LM(0.26, 0.45)) # idx pip 6
hand_left.append(LM(0.24, 0.37)) # idx dip 7
hand_left.append(LM(0.22, 0.30)) # idx tip 8
for _ in range(4): hand_left.append(LM(0.30, 0.60)) # mid 9-12 (curled)
for _ in range(4): hand_left.append(LM(0.32, 0.62)) # ring 13-16 (curled)
for _ in range(4): hand_left.append(LM(0.34, 0.64)) # pinky 17-20 (curled)

# Test angled hand (right arm raised like Messi, index angled 30 deg outward)
# Wrist at (0.7, 0.8), MCP at (0.72, 0.55), Tip at (0.78, 0.3)
hand_right = [LM(0.7, 0.8)] # wrist 0
for _ in range(4): hand_right.append(LM(0.68, 0.6)) # thumb 1-4
hand_right.append(LM(0.72, 0.55)) # idx mcp 5
hand_right.append(LM(0.74, 0.45)) # idx pip 6
hand_right.append(LM(0.76, 0.37)) # idx dip 7
hand_right.append(LM(0.78, 0.30)) # idx tip 8
for _ in range(4): hand_right.append(LM(0.70, 0.60)) # mid 9-12 (curled)
for _ in range(4): hand_right.append(LM(0.68, 0.62)) # ring 13-16 (curled)
for _ in range(4): hand_right.append(LM(0.66, 0.64)) # pinky 17-20 (curled)

print("Left hand pointing up:", is_pointing_up(hand_left))
print("Right hand pointing up:", is_pointing_up(hand_right))

# Test flat open hand (should be False)
hand_flat = [LM(0.5, 0.8)]
for _ in range(4): hand_flat.append(LM(0.4, 0.6))
for _ in range(4): hand_flat.append(LM(0.48, 0.3)) # idx extended
for _ in range(4): hand_flat.append(LM(0.50, 0.28)) # mid extended (higher than idx!)
for _ in range(4): hand_flat.append(LM(0.52, 0.30)) # ring extended
for _ in range(4): hand_flat.append(LM(0.54, 0.35)) # pinky extended
print("Flat open hand (should be False):", is_pointing_up(hand_flat))

# Test fist (should be False)
hand_fist = [LM(0.5, 0.8)]
for _ in range(20): hand_fist.append(LM(0.5, 0.65))
print("Fist (should be False):", is_pointing_up(hand_fist))
