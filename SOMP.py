import numpy as np
from numpy import linalg as LA


def omp(A, b, variance, *args):
    r = b
    samples = b.shape[1]
    cols = A.shape[1]
    rows = A.shape[0]
    if args == (): k = cols / 2
    else: k = args[0]
    A_reduced = np.empty((rows, 0))  # As Matrix
    X = np.zeros((cols, samples))
    s = []
    error_power = (LA.norm(r)**2) / cols
    while error_power >= variance and k > 0:
        Arr = abs(np.matmul(A.T, r))
        next_col = np.where(Arr == np.amax(Arr))[0]
        s.append(next_col.tolist()[0])
        A_reduced = np.column_stack((A_reduced, A[:, next_col]))
        Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
        X[s] = Xs
        b_hat = np.matmul(A, X)
        r = b - b_hat
        error_power = (LA.norm(r) ** 2 )/ cols
        k -= 1
    return X, error_power


from scipy.sparse import rand
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
rows = 20
cols = 50
k = 5
A = np.random.randn(rows, cols)
for i in range(cols):
    A[:,i] = A[:,i]/LA.norm(A[:,i])
n_samples = 4
X = rand(cols, n_samples, density=k / cols).todense()                           # <UPDATE: Specify the limits Uniform Dst> --> Resolved in Scratch 3
b = A * X
SNR = 30
variance = (LA.norm(b) ** 2 / rows) / (10 ** (SNR / 10))
n = np.random.randn(rows, 1) * np.sqrt(variance)
b_n = b + n
X_hat, resi = omp(A, b_n, variance, k)
# print(np.column_stack((X, X_hat)))
print(X, X.shape)
print(X_hat, X_hat.shape)
print(resi)
# plt.stem(X_hat, use_line_collection=True)
# plt.show()