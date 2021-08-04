import numpy as np
from numpy import linalg as LA
from scipy.sparse import rand
from OMP import omp


def learn_dict(rows, cols, non_zeros, b):
    samples = b.shape[1]

    # Randomly Initializing a dictionary

    dictionary = np.random.randn(rows, cols)
    for idx in range(cols):
        dictionary[:, idx] = dictionary[:, idx] / LA.norm(dictionary[:, idx])  # (20, 30)

    # Code book update stage

    error = 10

    while error >= 0.01:
        # Finding the sparse representations of b using this dict
        X_hat = np.empty((cols, 0))
        for idx in range(samples):
            new_X_hat = omp(non_zeros, dictionary, b[:, idx])
            X_hat = np.concatenate((X_hat, new_X_hat), axis=1)  # (30, 10)

        for kdx in range(cols):
            w_k = [idx for idx in range(samples) if X_hat[kdx, idx] != 0]
            X_hat_k = X_hat[kdx:kdx + 1, :].T[w_k]
            b_k = b[w_k]

            dictionary[:, kdx:kdx + 1] = 0
            E = b - np.matmul(dictionary, X_hat)  # (20,10)
            E_k = E.T[w_k].T                        # (20, |w_k|)

            U, S, V = LA.svd(E_k)
            dictionary[:, kdx:kdx + 1] = U[:, 0:1]

            if V.shape == (0, 0):
                X_hat_k = 0
            else:
                X_hat_k = V[:, 0:1] * S[0]

    print(dictionary)


if __name__ == '__main__':
    rows = 20
    cols = 30
    non_zeros = 5
    n_samples = 10

    # Dictionary
    A = np.random.randn(rows, cols)
    for i in range(cols):
        A[:, i] = A[:, i] / LA.norm(A[:, i])

    # Expected Sparse Signals
    X = rand(cols, n_samples,
             density=non_zeros / cols).todense()

    # Y
    b = A * X
    SNR = 50
    variance = (LA.norm(b) ** 2 / rows * n_samples) / (10 ** (SNR / 10))
    n = np.random.randn(rows, 1) * np.sqrt(variance)
    b_n = b + n

    # Testing

    learn_dict(rows, cols, non_zeros, b)
