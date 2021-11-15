import numpy as np
from numpy import linalg as LA


def omp(dictionary, signal_matrix, non_zeros):
    r = signal_matrix
    col = dictionary.shape[1]
    row = dictionary.shape[0]
    sparse_rep = np.zeros((col, 1))
    support = []

    while non_zeros:
        Arr = abs(np.matmul(dictionary.T, r))
        next_col = np.argmax(Arr)
        support.append(next_col)
        # support = sorted(support)
        sparse_rep[support] = np.matmul((LA.pinv(dictionary[:, support])), signal_matrix)
        signal_matrix_hat = np.matmul(dictionary, sparse_rep)
        r = signal_matrix - signal_matrix_hat
        non_zeros -= 1
    return sparse_rep
