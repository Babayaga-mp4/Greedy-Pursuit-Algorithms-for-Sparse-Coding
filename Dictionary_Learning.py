import numpy as np
from numpy import linalg as LA
from scipy.sparse import rand
from OMP import omp


def learn_dict(rows, cols, non_zeros, b):
    samples = b.shape[1]
    X_hat = np.empty((cols, 0))

    # Randomly Initializing a dictionary

    dictionary = np.random.randn(rows, cols)
    for idx in range(cols):
        dictionary[:, idx] = dictionary[:, idx] / LA.norm(dictionary[:, idx])   # (20, 30)

    # Finding the sparse representations of b using this dict

    for idx in range(samples):
        new_X_hat = omp(non_zeros, dictionary, b[:, idx])
        X_hat = np.concatenate((X_hat, new_X_hat), axis=1)          # (30, 10)

    # Code book update stage for a single atom. [Loop it up after one iteration]
    error = 10
    while error >= 0.01:
        for kdx in range(cols):

            w_k = [idx for idx in range(samples) if X_hat[kdx,idx] != 0]      # Gives the indices of the atoms that
            # use atom kdx

            # Defining Omega_k matrix for Atom kdx

            omega_k = np.zeros((samples, len(w_k)))   # [samples, len(w_k)]

            for idx in range(len(w_k)):
                omega_k[w_k[idx], idx] = 1

            # Define X_hat_k, b_k and E_k

            X_hat_k = np.matmul(X_hat[idx:idx+1, :], omega_k)
            b_k = np.matmul(b, omega_k)

            # Compute E

            temp = 0
            for idx in range(cols-1):
                if idx != 0:
                    temp += np.matmul(dictionary[:, idx: idx+1], X_hat[idx:idx+1, :])
            E = b - temp                    # (20,10)
            E_k = np.matmul(E, omega_k)     # (20, |w_k|)

            U, S, V = LA.svd(E_k)

            dictionary[:, idx:idx+1] = U[:, 0:1]
            X_hat_k = V[:, 0:1]*S[0]

        # Compute the error




# Share with:



rows = 20
cols = 30
non_zeros = 6
n_samples = 10

# Dictionary
A = np.random.randn(rows, cols)
for i in range(cols):
    A[:, i] = A[:, i] / LA.norm(A[:, i])

# Expected Sparse Signals
X = rand(cols, n_samples,
         density=non_zeros / cols).todense()  # <UPDATE: Specify the limits Uniform Dst> --> Resolved in Scratch 3

# Y
b = A * X
SNR = 30
variance = (LA.norm(b) ** 2 / rows) / (10 ** (SNR / 10))
n = np.random.randn(rows, 1) * np.sqrt(variance)
b_n = b + n

# Testing

learn_dict(rows, cols, non_zeros, b)
