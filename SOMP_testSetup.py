import numpy as np
from numpy import linalg as LA
import random
import matplotlib.pyplot as plt
from SOMP_Noise import somp


rows = 20
cols = 50
k = 3
A = np.random.randn(rows, cols)
for i in range(cols):
    A [:, i] = A [:, i] / LA.norm(A [:, i])

n_samples, abs_error = 5, []

for counter in range(0, 50, 5):
    error = 0
    SNR = counter

    for cdx in range(10):

        X = np.zeros((cols, n_samples))
        # dense = np.random.randn(k, n_samples)
        # print(dense)
        track = []
        idx = 0

        while idx < k:
            rand_ind = random.randint(0, cols - 1)
            if rand_ind not in track:
                X [rand_ind, :] = np.random.rand(1, n_samples)  # Generate a random row vector
                idx += 1
                track.append(rand_ind)

        # print(X)

        B = np.matmul(A, X)
        variance = (LA.norm(B) ** 2 / rows * n_samples) / (10 ** (SNR / 10))
        n = np.random.randn(rows, 1) * np.sqrt(variance)
        B_n = B + n
        X_hat = somp(A, B_n, variance, k)
        # print(X)
        # print(X_hat, X_hat.shape)
        B_hat = np.matmul(A, X_hat)
        error += ((LA.norm(B - B_hat)) ** 2) / (n_samples * 1000 * rows)
    abs_error.append(error)

plt.plot([idx for idx in range(0, 50, 5)], abs_error)
plt.show()
