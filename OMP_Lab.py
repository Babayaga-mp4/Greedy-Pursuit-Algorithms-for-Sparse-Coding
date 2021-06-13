import numpy as np
from numpy import linalg as LA

# Threshold
# def omp_1(A, b):
#     r = b  # Residual of k-1
#     i = 0  # Counter
#     cols = A.shape[1]
#     rows = A.shape[0]
#     A_reduced = np.ones((rows, 1))  # As Matrix
#     A_reduced = np.delete(A_reduced, 0, 1)
#     X = np.zeros((cols, 1))
#     s = []
#     threshold = 0.05 * (LA.norm(b))
#     error = LA.norm(r)
#     while error >= threshold:
#         Arr = abs(np.matmul(A.T, r))
#         next_col = np.where(Arr == np.amax(Arr))[0]
#         s.append(next_col.tolist()[0])
#         A_reduced = np.column_stack((A_reduced, A[:, next_col]))
#         Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
#         counter = 0
#
#         for index in s:
#             X[index] = Xs[counter]  # Avoid the FOR
#             counter += 1
#
#         # print('Sparse version:', X)
#         b_hat = np.matmul(A, X)
#         # print(b_hat.shape)
#         r = b - b_hat
#         error = LA.norm(r)  # Update
#         # print(r)
#         # print('In the loop:', np.max(r))
#         # if error < 0.0001: break                                                                    #Update
#         i += 1
#     return X

# Non Zeros

# def omp_2 (k, A, b):
#     r_prev = b                                                           #Residual of k-1
#     i = 0                                                                #Counter
#     cols = A.shape[1]
#     rows = A.shape[0]
#     A_reduced = np.ones((rows,1))                                        #As Matrix
#     A_reduced = np.delete(A_reduced,0,1)
#     X = np.zeros((cols, 1))
#     s = []
#     while i < k:
#         Arr = abs(np.matmul(A.T, r_prev ))
#         next_col = np.where(Arr == np.amax(Arr))[0]
#         s.append(next_col.tolist()[0])
#         A_reduced = np.column_stack((A_reduced, A[:, next_col]))
#         Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
#         counter = 0
#
#         for index in s:
#             X[index] = Xs[counter]                                                                      #Avoid the FOR
#             counter += 1
#
#         b_hat = np.matmul(A, X)
#         r_prev = b - b_hat
#         i += 1
#     return X

# Difference
# def omp_3(A, b):
#     r = b  # Residual of k-1
#     i = 0  # Counter
#     cols = A.shape[1]
#     rows = A.shape[0]
#     A_reduced = np.ones((rows, 1))  # As Matrix
#     A_reduced = np.delete(A_reduced, 0, 1)
#     X = np.zeros((cols, 1))
#     s = []
#     # threshold = (LA.norm(b))
#     error = LA.norm(r)
#     while i <= cols / 2:
#         Arr = abs(np.matmul(A.T, r))
#         next_col = np.where(Arr == np.amax(Arr))[0]
#         s.append(next_col.tolist()[0])
#         A_reduced = np.column_stack((A_reduced, A[:, next_col]))
#         Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
#         counter = 0
#
#         for index in s:
#             X[index] = Xs[counter]  # Avoid the FOR
#             counter += 1
#
#         # print('Sparse version:', X)
#         b_hat = np.matmul(A, X)
#         # print(b_hat.shape)
#         r_prev = r
#         r = b - b_hat
#         error = abs(LA.norm(r) - LA.norm(r_prev))  # Update
#         # print(r)
#         # print('In the loop:', np.max(r))
#         if error < 0.001: break  # Update
#         i += 1
#     return X


# Testing Program

import pyarma as pa
def omp_4(A, b, variance):
    r = b
    cols = A.n_cols
    rows = A.n_rows
    A_reduced = pa.mat(rows, 0, pa.fill.none)
    print(A_reduced)
    X_hat = pa.mat(cols, 1, pa.fill.zeros)
    s = []
    error_power = (pa.norm(r) ** 2)/cols
    while error_power >= variance:
        next_col = pa.index_max(A.t() * r)[0]
        s.append(next_col)
        A_reduced = pa.join_rows(A_reduced, A[:, next_col])
        Xs = pa.pinv(A_reduced) * b
        counter = 0
        for index in s:
            X_hat[index] = Xs[counter]  # Avoid the FOR
            counter += 1

        b_hat = (A * X_hat)
        r = b - b_hat
        error_power = (pa.norm(r)**2)/cols
    return X_hat
from scipy.sparse import rand
import matplotlib.pyplot as plt

rows = 20
cols = 50
k = 6
A = pa.mat(rows, cols, pa.fill.randu)
X = pa.mat(rand(cols, 1, density=k/cols).todense())                                 # <UPDATE: Specify the limits Uniform Dst> --> Resolved in Scratch 3
b = A * X
SNR = 20
variance = (pa.norm(b)**2/rows)/(10**(SNR/10))
n = pa.mat(rows, 1, pa.fill.randn)*variance
b_n = b + n
X_hat_4 = omp_4(A, b_n, variance)
print(pa.nonzeros(X_hat_4), pa.nonzeros(X))
plt.stem(X_hat_4)
plt.show()
plt.stem(X)
plt.show()