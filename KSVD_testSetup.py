import numpy as np
from numpy import linalg as LA
from KSVD import ksvd

rows = 30
cols = 60
non_zeros = 4
n_samples = 4000
for SNR in range(1):
    dictionary = np.random.randn(rows, cols)
    for idx in range(cols):
        dictionary[:, idx] = dictionary[:, idx] / (LA.norm(dictionary[:, idx]))  # (20, 30)

    X = np.zeros((cols, n_samples))
    X[:non_zeros,:] = np.random.randn(non_zeros, n_samples)

    for idx in range(n_samples):
        X[:, idx] = X[np.random.permutation(cols), idx]

    Y = np.matmul(dictionary, X)
    variance = 0.1
    n = np.random.randn(rows, n_samples) * np.sqrt(variance)
    Y_n = Y + n

    learned_dict, X_hat = ksvd(rows, cols, Y_n, variance, dictionary)




