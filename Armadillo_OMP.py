from pyarma import *


def omp(A, b, variance):
    r = b
    cols = A.n_cols
    rows = A.n_rows
    A_reduced = mat(rows, 0, fill.none)
    X_hat = mat(cols, 1, fill.zeros)
    s = []
    error_power = pow(normalise(r), 2) / cols
    print(error_power >= variance)
    while error_power >= variance:
        next_col = index_max(A.t() * r)[0]
        s.append(next_col)
        A_reduced = join_rows(A_reduced, A[:, next_col])
        Xs = pinv(A_reduced) * b
        # counter = 0
        alpha = mat(s)
        X_hat[alpha] = Xs
        X_hat.print()
        # for index in s:
        #     X_hat[index] = Xs[counter]
        #     counter += 1

        b_hat = (A * X_hat)
        r = b - b_hat
        error_power = (norm(r) ** 2) / cols
    return X_hat


from scipy.sparse import rand
import matplotlib.pyplot as plt

rows = 20
cols = 50
k = 6
A = mat(rows, cols, fill.randn)
A = normalise(A, 2, 1)
X = mat(rand(cols, 1, density=k / cols).todense())  # <UPDATE: Specify the limits Uniform Dst> --> Resolved in Scratch 3
b = A * X
SNR = 2
variance = (norm(b) ** 2 / rows) / (10 ** (SNR / 10))
n = mat(rows, 1, fill.randn) * variance
b_n = b + n
X_out = omp(A, b_n, variance)
# join_rows(X, X_out).print()
# plt.stem(X_hat)
# plt.show()
# plt.stem(X)
# plt.show()
