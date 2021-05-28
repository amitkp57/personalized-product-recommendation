import math

import numpy as np


def copy_values(X, X_hat):
    n, m = X.shape
    for i in range(n):
        for j in range(m):
            if not math.isnan(X[i][j]):
                X_hat[i][j] = X[i][j]
    return X_hat


def map_to_rating_values(X):
    X = np.array(map(lambda val: 1 if val <= 1.5 else val, X))
    X = np.array(map(lambda val: 2 if 1.5 < val <= 2.5 else val, X))
    X = np.array(map(lambda val: 3 if 2.5 < val <= 3.5 else val, X))
    X = np.array(map(lambda val: 4 if 3.5 < val <= 4.5 else val, X))
    X = np.array(map(lambda val: 5 if val > 4.5 else val, X))
    return X
