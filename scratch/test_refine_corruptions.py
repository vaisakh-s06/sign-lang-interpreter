import sys
sys.path.insert(0, '.')

import pickle
import numpy as np
import math

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)
data = d['data']
labels = d['labels']

with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)
model = model_obj['model']

# Import refine_prediction from app
from app import refine_prediction, LABELS_DICT

print("Checking which letters get overridden or corrupted by refine_prediction:")
corrupted_counts = {}

for i, (sample, lbl) in enumerate(zip(data, labels)):
    c_int = int(str(lbl))
    feat_array = np.asarray(sample, dtype=np.float32).reshape(1, -1)
    proba = model.predict_proba(feat_array)[0]
    best_idx = int(np.argmax(proba))
    class_id = int(str(model.classes_[best_idx]))
    conf = float(proba[best_idx])
    
    # What does refine_prediction do?
    ref_id, ref_conf = refine_prediction(class_id, conf, sample, proba, model)
    
    if class_id == c_int and ref_id != c_int:
        name = LABELS_DICT.get(c_int, str(c_int))
        hijacked_to = LABELS_DICT.get(ref_id, str(ref_id))
        corrupted_counts[name] = corrupted_counts.get(name, {})
        corrupted_counts[name][hijacked_to] = corrupted_counts[name].get(hijacked_to, 0) + 1

print("Corruptions directly from refine_prediction on clean dataset:")
for name, dests in corrupted_counts.items():
    print(f"  Letter {name} was hijacked to: {dests}")

if not corrupted_counts:
    print("  No corruptions on 100% confidence dataset.")

# Now test what happens when confidence is slightly lower (e.g. 0.65 or 0.60):
print("\nTesting what happens when confidence is 0.65 (typical real-life video):")
low_conf_hijacked = {}
for i, (sample, lbl) in enumerate(zip(data, labels)):
    c_int = int(str(lbl))
    ref_id, ref_conf = refine_prediction(c_int, 0.65, sample, None, None)
    if ref_id != c_int:
        name = LABELS_DICT.get(c_int, str(c_int))
        hijacked_to = LABELS_DICT.get(ref_id, str(ref_id))
        low_conf_hijacked[name] = low_conf_hijacked.get(name, {})
        low_conf_hijacked[name][hijacked_to] = low_conf_hijacked[name].get(hijacked_to, 0) + 1

for name, dests in low_conf_hijacked.items():
    print(f"  Letter {name} hijacked to: {dests}")
