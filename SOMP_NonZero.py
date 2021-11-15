import numpy as np
from numpy import linalg as LA


def somp(dictionary, signal_matrix, non_zeros):
    r = signal_matrix
    samples = signal_matrix.shape [1]
    col = dictionary.shape [1]
    row = dictionary.shape [0]
    sparse_rep = np.zeros((col, samples))
    support = []

    while non_zeros:
        Arr = np.sum(abs(np.matmul(dictionary.T, r)), axis = 1)  # Sum along the rows
        next_col = np.argmax(Arr)
        support.append(next_col)
        # support = sorted(support)
        sparse_rep [support, :] = np.matmul((LA.pinv(dictionary [:, support])), signal_matrix)
        signal_matrix_hat = np.matmul(dictionary, sparse_rep)
        r = signal_matrix - signal_matrix_hat
        non_zeros -= 1
    return sparse_rep



