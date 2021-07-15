from pyarma import *


def omp(dictionary, image, variance):
    r = image
    samples = image.n_cols
    column = dictionary.n_cols
    row = dictionary.n_rows
    A_reduced = mat(row, 0, fill.none)
    X_hat = mat(column, samples, fill.zeros)
    s = []
    error_power = pow(normalise(r), 2) / column
    # print(error_power >= variance)
    while error_power >= variance:
        next_col = index_max(dictionary.t() * r)[0]
        # print(max(dictionary.t() * r))
        # print(next_col)
        s.append(next_col)
        A_reduced = join_rows(A_reduced, dictionary[:, next_col])
        Xs = pinv(A_reduced) * image
        # print(size(Xs))
        counter = 0
        # alpha = mat(s)
        # X_hat[alpha] = Xs
        # X_hat.print()
        for index in s:
            X_hat[index] = Xs[counter]
            counter += 1

        b_hat = (dictionary * X_hat)
        r = image - b_hat
        error_power = (norm(r) ** 2) / column
        # error_power = pow(normalise(r), 2) / column Study this statement
    return X_hat


from scipy.sparse import rand
import matplotlib.pyplot as plt

rows = 30
cols = 60
k = 6
A = mat(rows, cols, fill.randn)
A = normalise(A, 2, 1)
n_samples = 4
X = mat(rand(cols, n_samples, density=k / cols).todense())  # <UPDATE: Specify the limits Uniform Dst> --> Resolved in Scratch 3
b = A * X
SNR = 50
var = (norm(b) ** 2 / rows) / (10 ** (SNR / 10))
n = mat(rows, 1, fill.randn) * var
# print(size(var))
b_n = b + n
X_out = omp(A, b_n, var)
join_rows(X, X_out).print()
# plt.stem(X_hat)
# plt.show()
# plt.stem(X)
# plt.show()

# Study the difference between norm and normalise