import numpy as np
from numpy import linalg as LA
from OMP_Noise import omp
from scipy.sparse import linalg
import matplotlib.pyplot as plt


def ksvd(rows, cols, signal_matrix, input_variance, true_dict):
    samples = signal_matrix.shape[1]
    num_iter = 50
    recovered = []
    dictionary = np.random.randn(rows, cols)
    for idx in range(cols):
        dictionary[:, idx] = dictionary[:, idx] / (LA.norm(dictionary[:, idx]))

    X_hat = np.empty((cols, 0))
    for idx in range(samples):
        new_X_hat = omp(dictionary, signal_matrix[:, idx:idx + 1], input_variance)
        X_hat = np.concatenate((X_hat, new_X_hat), axis = 1)

    for iter_counter in range(num_iter):

        for kdx in range(cols):

            w_k = [idx for idx in range(samples) if X_hat[kdx, idx] != 0]

            if len(w_k) == 0:
                continue

            tmpCoeff = X_hat[:, w_k]
            tmpCoeff[kdx, :] = 0
            E = signal_matrix[:, w_k] - np.matmul(dictionary, tmpCoeff)

            dict_element, S, V = linalg.svds(E, 1)
            dictionary[:, kdx:kdx + 1] = dict_element
            X_hat[kdx, w_k] = S * np.mat(V)

        X_hat = np.empty((cols, 0))

        for idx in range(cols):
            dictionary[:, idx] = dictionary[:, idx] / (LA.norm(dictionary[:, idx]))

        for idx in range(samples):
            new_X_hat = omp(dictionary, signal_matrix[:, idx:idx + 1], input_variance)
            X_hat = np.concatenate((X_hat, new_X_hat), axis = 1)
        counter = 0

        for idx in range(cols):
            d = true_dict[:, idx]
            for jdx in range(cols):
                dt = dictionary[:, jdx]
                s = abs(np.matmul(d.T, dt))
                if s >= 0.8:
                    counter += 1
                    break
        print('iter_counter {}: {}'.format(iter_counter, counter))
        recovered.append(100 * (counter / cols))
    # print(recovered)
    # plt.plot([idx for idx in range(0, 50)], recovered)
    # plt.show()
    return dictionary, recovered
