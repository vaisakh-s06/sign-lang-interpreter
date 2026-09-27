import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = d['data']
labels = d['labels']

print(f"Total samples: {len(data)}")

# Classes to verify:
# G: 6, J: 9, M: 12, N: 13, R: 17
# Also confusing neighbours:
# H: 7, I: 8, U: 20, V: 21, A: 0, S: 18

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

targets = ['6', '7', '8', '9', '12', '13', '17', '20', '21']
for t in targets:
    indices = [i for i, lbl in enumerate(labels) if str(lbl) == t]
    if not indices:
        continue
    X_sub = np.array([data[i] for i in indices], dtype=np.float32)
    preds = model.predict(X_sub)
    probas = model.predict_proba(X_sub)
    confs = np.max(probas, axis=1)
    acc = np.mean(preds == t)
    mean_conf = np.mean(confs)
    print(f"Class {t:>2}: samples={len(indices)}, acc={acc*100:.1f}%, mean_conf={mean_conf*100:.1f}%")
