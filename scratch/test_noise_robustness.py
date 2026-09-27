import pickle, numpy as np

with open('model.p', 'rb') as f:
    clf = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = np.array(d['data'], dtype=np.float32)
labels = np.array([int(str(x)) for x in d['labels']])

names = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.'
}

# Test noise robustness on classes 6, 9, 12, 13, 17
# Add small Gaussian noise (std = 0.02, 0.04) to simulate slightly different hand size/angles/positions
for c in [6, 9, 12, 13, 17]:
    idx = (labels == c)
    c_samples = data[idx]
    
    print(f"\n--- Class {c:2d} ({names[c]}) Robustness ---")
    for noise_level in [0.01, 0.025, 0.04]:
        confidences = []
        correct = 0
        confusions = {}
        for s in c_samples:
            # 5 noisy variations per sample
            for _ in range(5):
                noisy_s = s + np.random.normal(0, noise_level, size=s.shape)
                # re-normalize
                xs = noisy_s[0::2]
                ys = noisy_s[1::2]
                mx, my = xs.min(), ys.min()
                sc = max(xs.max() - mx, ys.max() - my) or 1.0
                norm_s = []
                for x, y in zip(xs, ys):
                    norm_s.extend([(x-mx)/sc, (y-my)/sc])
                
                proba = clf.predict_proba([norm_s])[0]
                best_idx = np.argmax(proba)
                pred_c = int(str(clf.classes_[best_idx]))
                conf = proba[best_idx]
                confidences.append(conf)
                if pred_c == c:
                    correct += 1
                else:
                    confusions[names[pred_c]] = confusions.get(names[pred_c], 0) + 1
        
        total = len(c_samples) * 5
        avg_conf = np.mean(confidences)
        print(f"Noise {noise_level:5.3f}: Correct = {correct}/{total} ({correct/total*100:.1f}%), Avg Conf = {avg_conf*100:.1f}%, Confusions: {confusions}")
