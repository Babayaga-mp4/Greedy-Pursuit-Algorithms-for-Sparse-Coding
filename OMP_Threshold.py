import numpy as np
from numpy import linalg as LA


def omp(A, b, threshold):
    r = b
    i = 0
    cols = A.shape[1]
    rows = A.shape[0]
    A_reduced = np.empty((rows, 0))
    X = np.zeros((cols, 1))
    s = []
    error = LA.norm(r)
    while error >= threshold:
        Arr = abs(np.matmul(A.T, r))
        next_col = np.where(Arr == np.amax(Arr))[0]
        s.append(next_col.tolist()[0])
        A_reduced = np.column_stack((A_reduced, A[:, next_col]))
        Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
        X[s] = Xs
        b_hat = np.matmul(A, X)
        r = b - b_hat
        error = LA.norm(r)
        i += 1
    return X


# Testing Program


from scipy.sparse import rand
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
rows = 20
cols = 50
k = 6
A = np.random.randn(rows, cols)
X = rand(cols, 1, density=k / cols).todense()  # <UPDATE: Specify the limits Uniform Dst>
b = A * X
threshold = 0.05 * (LA.norm(b))
X_hat = omp(A, b, threshold)
print(np.column_stack((X, X_hat)))
plt.stem(X_hat, use_line_collection=True)
plt.show()
