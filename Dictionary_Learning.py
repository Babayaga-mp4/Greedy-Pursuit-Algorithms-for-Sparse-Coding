import numpy as np
from numpy import linalg as LA
from OMP_Noise import omp


def ksvd(rows, cols, b, input_variance):
    samples = b.shape[1]
    num_iter = 20

    # Randomly Initializing a dictionary

    dictionary = np.random.randn(rows, cols)
    for idx in range(cols):
        dictionary[:, idx] = dictionary[:, idx] / LA.norm(dictionary[:, idx])  # (20, 30)

    iter_counter = 0
    error = float('inf')  # Make Infinity

    while iter_counter < num_iter:

        # while error > c*input_variance:

        # Sparse Coding

        X_hat = np.empty((cols, 0))
        for idx in range(samples):
            new_X_hat = omp(dictionary, b[:, idx:idx+1], input_variance)
            X_hat = np.concatenate((X_hat, new_X_hat), axis = 1)  # (30, 10)

        # Dictionary Update

        for kdx in range(cols - 1):

            w_k = [idx for idx in range(samples) if X_hat[kdx, idx] != 0]

            if len(w_k) == 0:
                continue

            X_hat_k = X_hat[kdx:kdx+1, :].T[w_k]
            # b_k = b[w_k]
            dictionary[:, kdx:kdx + 1] = 0
            # E = b - np.matmul(dictionary, X_hat)  # (20,10)       # Check the coherence between E and E_k
            # E_k = E.T[w_k].T  # (20, |w_k|)

            E_k = b[:, w_k] - np.matmul(dictionary, X_hat_k)
            U, S, V = LA.svd(E_k)
            dictionary[:, kdx:kdx + 1] = U[:, 0:1]

            X_hat_k = V[:, 0:1] * S[0]
            X_hat[kdx:kdx + 1, :].T[w_k] = X_hat_k

            residual = b - np.matmul(dictionary, X_hat)
            error = (LA.norm(residual) ** 2) / samples

        iter_counter += 1
    # print(iter_counter)
    if error < input_variance: print('Yes', error, input_variance)
    return dictionary
