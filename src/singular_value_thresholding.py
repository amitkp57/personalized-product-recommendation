import numpy as np

from src.data_loader import create_dataset
from src.utils import copy_values


def matrix_completion(X, threshold=1, epsilon=0.01):
    """
    Lower rank matrix completion using iterative singular value thresholding
    :param X:
    :param threshold:
    :param epsilon:
    :return:
    """
    n, m = X.shape
    X_hat = np.zeros((n, m))
    X_hat = copy_values(X, X_hat)
    while True:
        X_hat_old = X_hat
        u, e, vh = np.linalg.svd(X_hat, full_matrices=True)
        # remove singular values less than threshold
        e = list(map(lambda val: val if val >= threshold else 0, e))
        E = np.zeros((n, m))
        E[0:min(m, n), 0:min(m, n)] = np.diag(e)
        X_hat = np.matmul(np.matmul(u, E), vh)
        X_hat = copy_values(X, X_hat)
        f_norm = np.linalg.norm(X_hat - X_hat_old)
        if f_norm < epsilon:
            break
    return X_hat


if __name__ == '__main__':
    X = create_dataset('Patio_Lawn_and_Garden_5').to_numpy(dtype=float)
    X_hat = matrix_completion(X, 50, 0.01)
    pass
