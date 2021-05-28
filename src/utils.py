import math


def copy_values(X, X_hat):
    n, m = X.shape
    for i in range(n):
        for j in range(m):
            if not math.isnan(X[i][j]):
                X_hat[i][j] = X[i][j]
    return X_hat


def map_to_rating_values(X):
    n, m = X.shape
    for i in range(n):
        for j in range(m):
            val = X[i][j]
            if val <= 1.5:
                X[i][j] = 1
            elif 1.5 < val <= 2.5:
                X[i][j] = 2
            elif 2.5 < val <= 3.5:
                X[i][j] = 3
            elif 3.5 < val <= 4.5:
                X[i][j] = 4
            else:
                X[i][j] = 5

    return X
