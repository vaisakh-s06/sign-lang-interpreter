import pickle, numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(str(x)) for x in d['labels']])

# 21 points: 0 is wrist
# 4: thumb tip, 8: index tip, 12: mid tip, 16: ring tip, 20: pinky tip
def print_hand_summary(c, name):
    c_data = data[labels == c]
    mean_feat = c_data.mean(axis=0)
    # mean_feat has (x, y) for each of 21 landmarks
    wrist = (mean_feat[0], mean_feat[1])
    th_tip = (mean_feat[8], mean_feat[9])
    idx_tip = (mean_feat[16], mean_feat[17])
    mid_tip = (mean_feat[24], mean_feat[25])
    ring_tip = (mean_feat[32], mean_feat[33])
    pky_tip = (mean_feat[40], mean_feat[41])
    print(f"\n=== Class {c:2d} ({name}) ===")
    print(f"  Wrist    : x={wrist[0]:.2f}, y={wrist[1]:.2f}")
    print(f"  Thumb tip: x={th_tip[0]:.2f}, y={th_tip[1]:.2f}")
    print(f"  Index tip: x={idx_tip[0]:.2f}, y={idx_tip[1]:.2f}")
    print(f"  Mid tip  : x={mid_tip[0]:.2f}, y={mid_tip[1]:.2f}")
    print(f"  Ring tip : x={ring_tip[0]:.2f}, y={ring_tip[1]:.2f}")
    print(f"  Pinky tip: x={pky_tip[0]:.2f}, y={pky_tip[1]:.2f}")

for c, n in [(6, 'G'), (7, 'H'), (8, 'I'), (9, 'J'), (12, 'M'), (13, 'N'), (17, 'R'), (20, 'U'), (21, 'V')]:
    print_hand_summary(c, n)
