import numpy as np
from numpy import linalg as LA


def omp(dictionary, signal_matrix, input_variance):
    # signal_matrix = np.mat(signal_matrix)
    r = signal_matrix
    col_ = dictionary.shape[1]
    row_ = dictionary.shape[0]
    sparse_rep = np.zeros((col_, 1))
    support = []

    error_power = float('inf')

    while error_power >= 1.15 * input_variance:
        Arr = abs(np.matmul(dictionary.T, r)).tolist()
        next_col = Arr.index(max(Arr))
        support.append(next_col)
        support = sorted(support)
        sparse_rep[support] = np.matmul((np.linalg.pinv(dictionary[:, support])), signal_matrix)
        signal_matrix_hat = np.matmul(dictionary, sparse_rep)
        r = signal_matrix - signal_matrix_hat
        error_power = (LA.norm(r) ** 2) / row_
    return sparse_rep


def omp2(dictionary, signal_matrix, input_variance):
    [n, P] = np.shape(signal_matrix)
    [n, K] = np.shape(dictionary)
    error_goal = 1.15*input_variance
    max_coeffs = n/2
    sparse_rep = np.zeros((dictionary.shape[1], signal_matrix.shape[1]))
    for kdx in range(P):
        a = []
        x = signal_matrix[:, kdx]
        r = x
        indx = []
        r_norm = np.sum(r**2)
        j = 0
        while r_norm > error_goal and j < max_coeffs:
            j += 1
            proj = abs(np.matmul(dictionary.T, r)).tolist()
            pos = proj.index(max(proj))
            indx.append(pos)
            # a_pinv = np.matmul(dictionary[:, indx[:j]], LA.inv(np.matmul(dictionary[:, indx[:j]], dictionary[:, indx[:j]].T)))
            # a = a_pinv*x
            a = np.matmul(LA.pinv(dictionary[:, indx[:j]]),x)
            # print(dictionary[:, indx[:j]].shape, a.shape)
            r = x - np.matmul(dictionary[:, indx[:j]], a.T)
            r_norm = np.sum(r**2)


        if len(indx) == 0:
            sparse_rep[indx, kdx] = a


    return sparse_rep
