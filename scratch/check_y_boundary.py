import pickle
import numpy as np

d = pickle.load(open('data.pickle', 'rb'))
data = np.array(d['data'])

for idx in [2349, 2429, 2430]:
    s = data[idx]
    xs = s[0::2]
    ys = s[1::2]
    dist = np.sqrt((xs[4] - xs[5])**2 + (ys[4] - ys[5])**2)
    print(f"Sample {idx}: xs[4]={xs[4]:.3f}, dist={dist:.3f}, xs[4]-xs[2]={xs[4]-xs[2]:.3f}")

# Also check all samples of J (9):
labels = np.array([int(x) for x in d['labels']])
samples_j = data[labels == 9]
xs_j = samples_j[:, 0::2]
ys_j = samples_j[:, 1::2]
dists_j = np.sqrt((xs_j[:, 4] - xs_j[:, 5])**2 + (ys_j[:, 4] - ys_j[:, 5])**2)
print(f"J (9): xs[4] max={np.max(xs_j[:, 4]):.3f}, dist max={np.max(dists_j):.3f}, xs[4]-xs[2] max={np.max(xs_j[:, 4] - xs_j[:, 2]):.3f}")

# Also check all samples of Y (24):
samples_y = data[labels == 24]
xs_y = samples_y[:, 0::2]
ys_y = samples_y[:, 1::2]
dists_y = np.sqrt((xs_y[:, 4] - xs_y[:, 5])**2 + (ys_y[:, 4] - ys_y[:, 5])**2)
print(f"Y (24): xs[4] min={np.min(xs_y[:, 4]):.3f}, dist min={np.min(dists_y):.3f}, xs[4]-xs[2] min={np.min(xs_y[:, 4] - xs_y[:, 2]):.3f}")
