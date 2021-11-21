import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from MOD import mod
from KSVD import ksvd

rows = 30
cols = 60
non_zeros = 4
n_samples = 4000
recovered, error = [], []


for SNR in range(1):
    dictionary = np.random.randn(rows, cols)
    for idx in range(cols):
        dictionary[:, idx] = dictionary[:, idx] / LA.norm(dictionary[:, idx])  # (20, 30)

    X = np.zeros((cols, n_samples))
    X[:non_zeros, :] = np.random.randn(non_zeros, n_samples)

    for idx in range(n_samples):
        X[:, idx] = X[np.random.permutation(cols), idx]

    variance = 0.1
    Y = np.matmul(dictionary, X)

    n = np.random.randn(rows, n_samples) * np.sqrt(variance)
    Y_n = Y + n

    learned_dict_KSVD, recovered_ksvd = ksvd(rows, cols, Y_n, variance, dictionary)
    learned_dict_mod, recovered_mod = mod(rows, cols, Y_n, variance, dictionary)


plt.plot([idx for idx in range(0, 50)], recovered_mod, 'red')
plt.plot([idx for idx in range(0, 50)], recovered_ksvd, 'black')

plt.show()
