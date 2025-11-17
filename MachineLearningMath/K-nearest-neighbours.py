import numpy as np
from collections import Counter
class KNN:
    def __init__(self, k=5, weighted=False, eps=1e-9):
        self.k = k
        self.weighted = weighted
        self.eps = eps

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)

    def dist(self, A, b):
        diff = A - b
        return np.sqrt((diff * diff).sum(axis=1))  # Euclidean

    def predict(self, Xq):
        Xq = np.asarray(Xq, dtype=float)
        preds = []
        for i in Xq:
            d = self.dist(self.X, i)                
            idx = np.argpartition(d, self.k)[:self.k]
            if not self.weighted:
                vote = Counter(self.y[idx]).most_common(1)[0][0]
            else:
                w = 1.0 / (d[idx] + self.eps)
                scores = {}
                for cls in (0, 1):
                    scores[cls] = w[self.y[idx] == cls].sum()
                vote = 1 if scores[1] >= scores[0] else 0
            preds.append(vote)
        return np.array(preds)
    
X_train = np.array([
    [0.2, 0.1, 0.5, 0.3, 0.4, 0.7, 0.6, 0.8, 0.9, 0.3],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9],
    [0.5, 0.4, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0.1, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.2, 0.1],
    [0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9]
])

# corresponding labels
y_train = np.array([1, 0, 1, 0, 0])

X_test = np.array([
    [0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3]
])
    
scratch = KNN(k=5, weighted=False)
scratch.fit(X_train, y_train)
y_pred_s = scratch.predict(X_test)
y_pred_s