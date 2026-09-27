import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)
classes = np.array([int(c) for c in model.classes_])

def mirror_features(data_aux):
    xs = np.array(data_aux[0::2])
    ys = np.array(data_aux[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-4:
        scale = 1.0
    out = []
    for x, y in zip((xs_m - min_x) / scale, (ys - min_y) / scale):
        out.extend([float(x), float(y)])
    return out

# Let's inspect predictions for G (6)
samples_g = data[labels == 6]
p_g_orig = model.predict_proba(samples_g)
preds_g_orig = classes[np.argmax(p_g_orig, axis=1)]
print(f"G (6) direct accuracy: {np.mean(preds_g_orig == 6)*100:.1f}%")

# Let's inspect predictions for T (19)
samples_t = data[labels == 19]
p_t_orig = model.predict_proba(samples_t)
preds_t_orig = classes[np.argmax(p_t_orig, axis=1)]
print(f"T (19) direct accuracy: {np.mean(preds_t_orig == 19)*100:.1f}%")

# Now let's check what happens with DUAL CHIRALITY when a user shows G or T with their left hand:
samples_g_left = np.array([mirror_features(s) for s in samples_g])
p_gl_o = model.predict_proba(samples_g_left)
p_gl_m = model.predict_proba(np.array([mirror_features(s) for s in samples_g_left]))
p_gl_comb = np.maximum(p_gl_o, p_gl_m)
preds_gl = classes[np.argmax(p_gl_comb, axis=1)]
print(f"G (6) left hand (dual chirality) accuracy: {np.mean(preds_gl == 6)*100:.1f}%")

samples_t_left = np.array([mirror_features(s) for s in samples_t])
p_tl_o = model.predict_proba(samples_t_left)
p_tl_m = model.predict_proba(np.array([mirror_features(s) for s in samples_t_left]))
p_tl_comb = np.maximum(p_tl_o, p_tl_m)
preds_tl = classes[np.argmax(p_tl_comb, axis=1)]
print(f"T (19) left hand (dual chirality) accuracy: {np.mean(preds_tl == 19)*100:.1f}%")
