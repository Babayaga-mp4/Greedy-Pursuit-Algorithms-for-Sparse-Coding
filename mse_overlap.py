import numpy as np
from numpy import linalg as LA
import random
from OMP_Noise import omp
from SOMP_Noise import somp
import matplotlib.pyplot as plt

rows = 20
cols = 50
k = 3
n_samples = 5
N = 1000
A = np.random.randn(rows, cols)
for i in range(cols):
    A[:, i] = A[:, i] / LA.norm(A[:, i])

X = np.zeros((cols, n_samples * N))

for idx in range(0, N, n_samples):
    index = random.choices(range(0,cols), k=k)
    X[index, idx: idx+n_samples] = np.random.rand(k, n_samples)

Y = np.matmul(A, X)

abs_error_omp, abs_error_somp = [], []

for SNR in range(0, 55, 5):
    variance = LA.norm(Y) ** 2 / (N * n_samples*10 ** (SNR / 10))
    n = np.random.randn(rows, N*n_samples) * np.sqrt(variance)
    Y_n = Y + n

    X_hat_omp = np.empty((cols, 0))

    for jdx in range(N*n_samples):
        X_hat_omp_ = omp(A, Y_n[:, jdx:jdx + 1], variance)
        X_hat_omp = np.concatenate((X_hat_omp, X_hat_omp_), axis = 1)

    Y_hat_omp = np.matmul(A, X_hat_omp)
    abs_error_omp.append(((LA.norm(Y - Y_hat_omp)) ** 2) / (N * n_samples))

    X_hat_somp = np.empty((cols, 0))

    for kdx in range(0, N*n_samples, n_samples):
        X_hat_somp_ = somp(A, Y_n[:, kdx:kdx+n_samples], variance)
        X_hat_somp = np.concatenate((X_hat_somp, X_hat_somp_), axis = 1)

    Y_hat_somp = np.matmul(A, X_hat_somp)
    print(X_hat_somp.shape)
    abs_error_somp.append(((LA.norm(Y - Y_hat_somp)) ** 2) / (N * n_samples))

plt.plot([idx for idx in range(0, 55, 5)], abs_error_somp, 'red')
plt.plot([idx for idx in range(0, 55, 5)], abs_error_omp, 'black')
plt.show()