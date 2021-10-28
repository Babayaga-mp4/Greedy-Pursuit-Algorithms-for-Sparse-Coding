import numpy as np
from numpy import linalg as LA

# 1st iteration of OMP
def omp (k, A, b):
    r_prev = b
    i = 0
    cols = A.shape[1]
    rows = A.shape[0]
    A_reduced = np.empty((rows, 0))
    X = np.zeros((cols, 1))
    s = []
    while i < k:
        Arr = abs(np.matmul(A.T, r_prev ))
        next_col = np.where(Arr == np.amax(Arr))[0]
        s.append(next_col.tolist()[0])
        A_reduced = np.column_stack((A_reduced, (A[:, next_col])))
        Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
        X[s] = Xs
        b_hat = np.matmul(A, X)
        r_prev = b - b_hat
        i += 1
    return X


#Testing Program


from scipy.sparse import rand
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
rows = 100
cols = 500
A = np.random.randn(rows, cols)
for i in range(cols):
    A[:,i] = A[:,i] / LA.norm(A[:,i])

abs_error = []
for counter in range(20):
    error = 0
    for idx in range(1000):
        k = counter
        X = rand(cols, 1, density= k / cols).todense()                           # <UPDATE: Specify the limits Uniform Dst> --> Resolved in Scratch 3
        b = np.matmul(A, X)
        X_hat = omp(k, A, b)
        b_hat = np.matmul(A, X_hat)
        # print(np.column_stack((X, X_hat)))
        error += (LA.norm(b - b_hat) ** 2) / 1000*rows
        # print(non_zeros)
    abs_error.append(error)
plt.plot([idx for idx in range(20)], abs_error)
plt.show()

