import numpy as np
from scipy.sparse import rand
import matplotlib.pyplot as plt
from numpy import linalg as LA
from OMP_Noise import omp

rows = 20
cols = 50
A = np.random.randn(rows, cols)
for i in range(cols):
    A[:, i] = A[:, i] / LA.norm(A[:, i])

k, abs_error = 3, []
for counter in range(0, 50, 5):
    error = 0
    for idx in range(1000):
        X = rand(cols, 1,
                 density = k / cols).todense()
        b = np.matmul(A, X)
        SNR = 40
        variance = (LA.norm(b) ** 2 / rows) / (10 ** (SNR / 10))
        n = np.random.randn(rows, 1) * np.sqrt(variance)
        b_n = b + n
        X_hat = omp(A, b_n, 1.2*variance)
        b_hat = np.matmul(A, X_hat)
        error += (LA.norm(b - b_hat) ** 2) / (1000 * rows)
    abs_error.append(error)
plt.plot([idx for idx in range(0, 50, 5)], abs_error)
plt.show()
