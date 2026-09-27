import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

# Check finger tips vs PIPs for index (8 vs 6) and middle (12 vs 10)
for target_lbl, target_name in [(3, 'D'), (17, 'R'), (20, 'U'), (21, 'V'), (10, 'K'), (25, 'Z')]:
    mask = (labels == target_lbl)
    samples = data[mask]
    
    # In normalized coords:
    # xs: 0, 2, ..., 40
    # ys: 1, 3, ..., 41
    # Index tip is landmark 8 -> y is index 8*2+1 = 17, PIP is landmark 6 -> y is index 6*2+1 = 13
    # Middle tip is landmark 12 -> y is index 12*2+1 = 25, PIP is landmark 10 -> y is index 10*2+1 = 21
    
    idx_tip_y = samples[:, 17]
    idx_pip_y = samples[:, 13]
    mid_tip_y = samples[:, 25]
    mid_pip_y = samples[:, 21]
    
    idx_up = idx_tip_y < idx_pip_y
    mid_up = mid_tip_y < mid_pip_y
    
    print(f"Class {target_name} ({target_lbl}): N={len(samples)}")
    print(f"  Index finger pointing UP: {np.mean(idx_up)*100:.1f}%")
    print(f"  Middle finger pointing UP: {np.mean(mid_up)*100:.1f}%")
    
    # Check fingertip distance between index (8) and middle (12)
    idx_tip_x = samples[:, 16]
    mid_tip_x = samples[:, 24]
    
    # Crossing metric:
    # Knuckle dx: xs[5] - xs[9] -> samples[:, 10] - samples[:, 18]
    # Tip dx: xs[8] - xs[12] -> samples[:, 16] - samples[:, 24]
    knuckle_dx = samples[:, 10] - samples[:, 18]
    tip_dx = samples[:, 16] - samples[:, 24]
    cross = knuckle_dx * tip_dx
    print(f"  Cross metric mean: {np.mean(cross):.5f}, min: {np.min(cross):.5f}, max: {np.max(cross):.5f}")
    print(f"  Tip horizontal distance |x8 - x12| mean: {np.mean(np.abs(tip_dx)):.4f}")
