import numpy as np
from numpy import linalg as LA


def omp(dictionary, signal_matrix, input_variance):
    r = signal_matrix
    col = dictionary.shape[1]
    row = dictionary.shape[0]
    sparse_rep = np.zeros((col, 1))
    support = []

    error_power = float('inf')

    while error_power >= 1.15 * input_variance:
        Arr = abs(np.matmul(dictionary.T, r))
        next_col = np.argmax(Arr)
        support.append(next_col)
        support = sorted(support)
        sparse_rep[support] = np.matmul((np.linalg.pinv(dictionary[:, support])), signal_matrix)
        signal_matrix_hat = np.matmul(dictionary, sparse_rep)
        r = signal_matrix - signal_matrix_hat
        error_power = (LA.norm(r) ** 2) / row
    return sparse_rep


