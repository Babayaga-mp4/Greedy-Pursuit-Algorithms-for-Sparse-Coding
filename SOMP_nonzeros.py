import numpy as np
from numpy.linalg import norm
# 1st iteration of OMP
def omp (k, A, b):
    r_prev = b
    i = 0
    samples = b.shape[1]
    cols = A.shape[1]
    rows = A.shape[0]
    A_reduced = np.empty((rows, 0))
    X = np.zeros((cols, samples))
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


# from scipy.sparse import rand
# from matplotlib import pyplot
# import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# rows = 20
# cols = 50
# k = 6
# n_samples = 4
# A = np.random.randn(rows, cols)
# for i in range(cols):
#     A[:,i] = A[:,i] / norm(A[:,i])
# X = rand(cols, n_samples, density= k/cols).todense()                                    #<UPDATE: Specify the limits Uniform Dst> --> Resolved in scratch 3
# b = A * X
# X_hat = omp(k, A, b)
# # print(np.column_stack((X, X_hat)), len(X))
# print(X)
# print(X_hat)
# # TRY PYTHON EQUIVALENT OF STEM
# # pyplot.stem(X_hat, use_line_collection=True)
# # pyplot.show()
