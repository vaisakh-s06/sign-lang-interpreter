import pickle, numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(str(x)) for x in d['labels']])

# 1. R (17) vs U (20)
# 8 is index tip (x: 16, y: 17)
# 12 is mid tip   (x: 24, y: 25)
# 5 is index mcp  (x: 10, y: 11)
# 9 is mid mcp    (x: 18, y: 19)
r_data = data[labels == 17]
u_data = data[labels == 20]

# For R: index tip x vs mid tip x
# In standard right hand: index MCP (5) is to the right of Mid MCP (9) or vice versa
r_cross = r_data[:, 16] - r_data[:, 24] # idx_x - mid_x
u_cross = u_data[:, 16] - u_data[:, 24]
print(f"R (idx_x - mid_x) mean: {r_cross.mean():.3f} (min={r_cross.min():.3f}, max={r_cross.max():.3f})")
print(f"U (idx_x - mid_x) mean: {u_cross.mean():.3f} (min={u_cross.min():.3f}, max={u_cross.max():.3f})")

# 2. M (12) vs N (13) vs A (0)
# Check ring tip y (idx 33) vs mid tip y (idx 25)
m_data = data[labels == 12]
n_data = data[labels == 13]
a_data = data[labels == 0]

print(f"\nM Ring tip y: {m_data[:, 33].mean():.3f}, Mid tip y: {m_data[:, 25].mean():.3f}, Diff (Ring - Mid): {(m_data[:, 33] - m_data[:, 25]).mean():.3f}")
print(f"N Ring tip y: {n_data[:, 33].mean():.3f}, Mid tip y: {n_data[:, 25].mean():.3f}, Diff (Ring - Mid): {(n_data[:, 33] - n_data[:, 25]).mean():.3f}")
print(f"A Ring tip y: {a_data[:, 33].mean():.3f}, Mid tip y: {a_data[:, 25].mean():.3f}, Diff (Ring - Mid): {(a_data[:, 33] - a_data[:, 25]).mean():.3f}")

# 3. G (6) vs H (7)
# Check middle tip extension
g_data = data[labels == 6]
h_data = data[labels == 7]
# In G, index tip x (idx 16) vs wrist (idx 0), mid tip x (idx 24)
print(f"\nG Index tip x: {g_data[:, 16].mean():.3f}, Mid tip x: {g_data[:, 24].mean():.3f}")
print(f"H Index tip x: {h_data[:, 16].mean():.3f}, Mid tip x: {h_data[:, 24].mean():.3f}")
