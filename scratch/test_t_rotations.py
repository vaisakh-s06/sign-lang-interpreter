import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(x) for x in d['labels']], dtype=np.int32)

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

mask_t = (labels == 19)
samples_t = data[mask_t]
classes = np.array([int(c) for c in model.classes_])

print("Testing T with small rotations (-20 deg to +20 deg):")
for angle in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    preds = []
    for s in samples_t:
        xs = s[0::2]
        ys = s[1::2]
        cx, cy = np.mean(xs), np.mean(ys)
        xr = cos_a * (xs - cx) - sin_a * (ys - cy) + cx
        yr = sin_a * (xs - cx) + cos_a * (ys - cy) + cy
        min_x, max_x = np.min(xr), np.max(xr)
        min_y, max_y = np.min(yr), np.max(yr)
        scale = max(max_x - min_x, max_y - min_y)
        if scale < 1e-4: scale = 1.0
        f = []
        for x, y in zip((xr - min_x)/scale, (yr - min_y)/scale):
            f.extend([float(x), float(y)])
        
        f_m = mirror_features(f)
        p_o = model.predict_proba(np.array(f).reshape(1, -1))[0]
        p_m = model.predict_proba(np.array(f_m).reshape(1, -1))[0]
        p = np.maximum(p_o, p_m)
        pred = classes[np.argmax(p)]
        preds.append(pred)
    
    unique, counts = np.unique(preds, return_counts=True)
    dist = dict(zip(unique, counts))
    acc = dist.get(19, 0) / len(preds) * 100
    print(f"  Angle {angle:+3d} deg: T accuracy = {acc:.1f}%, preds: {dist}")
