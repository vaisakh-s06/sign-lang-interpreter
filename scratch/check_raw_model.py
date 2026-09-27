import pickle
import numpy as np

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

X_all = np.asarray(data, dtype=np.float32)
all_probas = model.predict_proba(X_all)
best_indices = np.argmax(all_probas, axis=1)

LABELS_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'Hello', 27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.',
    33: 'lional messi-the goat🐐'
}

unique_labels = sorted(list(set(labels)), key=lambda x: int(x) if x.isdigit() else 999)

print("RAW MODEL ACCURACY (WITHOUT refine_prediction):")
for lbl in unique_labels:
    indices = [i for i, l in enumerate(labels) if str(l) == str(lbl)]
    correct = 0
    confs = []
    for idx in indices:
        proba = all_probas[idx]
        pred_id = int(str(model.classes_[best_indices[idx]]))
        conf = float(proba[best_indices[idx]])
        confs.append(conf)
        if pred_id == int(lbl):
            correct += 1
    char_name = LABELS_DICT.get(int(lbl), f"Sign {lbl}")
    print(f"  Class {lbl:>2} ({char_name:<16}): {correct}/{len(indices)} ({correct/len(indices)*100:.1f}%), Mean Conf: {np.mean(confs)*100:.1f}%")
