import numpy as np

import src.data_loader as data_loader
from src.utils import copy_values, map_to_rating_values


def solve_V(X, U):
    """
    Given matrix X of shape (n, m),  U of shape (n, k), returns V of shape (m, k) such that X = U * transpose(V)
    :param X:
    :param U:
    :return:
    """
    n, m = X.shape
    k = U.shape[1]
    V = np.zeros((m, k))
    for i in range(m):
        column = X[:, i].flatten()
        indexes = np.argwhere(~np.isnan(column)).flatten()
        U_omega = U[indexes, :]
        y_omega = X[indexes, i]
        V[i, :] = np.linalg.lstsq(U_omega, y_omega)[0]
    return V


def alt_min(X, U_0, T):
    """
    Given X, initial values of U and number of iterations T, returns U, V such that X = U * transpose(V)
    :param X:
    :param U_0:
    :param T:
    :return:
    """
    U = U_0
    for _ in range(T):
        V = solve_V(X, U)
        U = solve_V(np.transpose(X), V)
    return U, V


def matrix_completion(X, T, k):
    """
    Given a matrix X with missing values, fills missing values in X
    :param X:
    :param T:
    :param k:
    :return:
    """
    X_hat = np.nan_to_num(X)
    U, e, V_t = np.linalg.svd(X_hat, full_matrices=True)
    U, V = alt_min(X, U[:, :k], T)
    X_filled = np.matmul(U, np.transpose(V))
    X_filled = copy_values(X, X_filled)
    X_filled = map_to_rating_values(X_filled)
    return X_filled


if __name__ == '__main__':
    T = 100
    k = 5
    X = data_loader.create_dataset('Patio_Lawn_and_Garden_5').to_numpy(dtype=float)
    # X_hat = matrix_completion(X, T, k)
    # print(X_hat)
    print(min(X.flatten()), max(X.flatten()))
