import numpy as np
from numpy import linalg as LA
from scipy import sparse as sc
from OMP_Noise import omp
import matplotlib.pyplot as plt


def mod(rows, cols, signal_matrix, input_variance, true_dict):
    samples = signal_matrix.shape[1]
    num_iter = 50
    dictionary = np.random.randn(rows, cols)
    for idx in range(cols):
        dictionary[:, idx] = dictionary[:, idx] / (LA.norm(dictionary[:, idx]))

    print(np.matmul(dictionary[:, 1], dictionary[:, 1].T))

    recovered = []

    for iter_count in range(num_iter):
        X_hat = np.empty((cols, 0))
        for idx in range(samples):
            new_X_hat = omp(dictionary, signal_matrix[:, idx:idx + 1], input_variance)
            X_hat = np.concatenate((X_hat, new_X_hat), axis = 1)
        X_hat_pinv = np.matmul(X_hat.T, LA.inv(
            np.matmul(X_hat, (X_hat.T + 1e-7 * sc.eye(X_hat.shape[1], X_hat.shape[0], dtype = 'int')))))
        dictionary_prev = dictionary
        dictionary = np.matmul(signal_matrix, X_hat_pinv)

        print((dictionary_prev == dictionary).all())

        for idx in range(cols):
            dictionary[:, idx] = dictionary[:, idx] / (LA.norm(dictionary[:, idx]))

        counter = 0
        for idx in range(cols):
            d = true_dict[:, idx]
            for jdx in range(cols):
                dt = dictionary[:, jdx]
                s = abs(np.matmul(d.T, dt))
                if s[0] >= 0.8:
                    counter += 1
                    break
        print('iter_counter {}: {}'.format(iter_count, counter))
        recovered.append(100 * (counter / cols))
    # print(recovered)
    # plt.plot([idx for idx in range(0, 50)], recovered)
    # plt.show()
    return dictionary, recovered
