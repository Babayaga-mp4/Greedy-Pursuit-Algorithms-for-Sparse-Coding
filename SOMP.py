import numpy as np
from numpy import linalg as LA


def somp(A, b, variance, *args):
    r = b
    samples = b.shape[1]
    cols = A.shape[1]
    rows = A.shape[0]
    if args == ():
        k = cols / 2
    else:
        k = args[0]
    A_reduced = np.empty((rows, 0))
    X = np.zeros((cols, samples))
    s = []
    error_power = (LA.norm(r) ** 2) / cols * samples
    while error_power >= variance and k > 0:
        Arr = abs(np.matmul(A.T, r))
        next_col = np.where(Arr == np.amax(Arr))[0]
        s.append(next_col.tolist()[0])
        A_reduced = np.column_stack((A_reduced, A[:, next_col]))
        Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
        X[s] = Xs
        b_hat = np.matmul(A, X)
        r = b - b_hat
        error_power = (LA.norm(r) ** 2) / cols * samples
        k -= 1
    print(error_power)
    return X


if __name__ == '__main__':
    from scipy.sparse import rand
    import random
    import matplotlib.pyplot as plt
    import warnings

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    rows = 20
    cols = 50
    k = 5
    A = np.random.randn(rows, cols)
    for i in range(cols):
        A[:, i] = A[:, i] / LA.norm(A[:, i])
    n_samples = 5
    # X = rand(cols, n_samples, density=k / cols).todense()
    X = np.zeros((cols, n_samples))
    dense = np.random.randn(k, n_samples)
    track = []
    idx = 0
    while idx < k:
        rand_ind = random.randint(0, cols)
        if rand_ind not in track:
            X[rand_ind, :] = dense[idx, :]
            idx += 1
            track.append(rand_ind)

    B = np.matmul(A, X)
    SNR = 30
    variance = (LA.norm(B) ** 2 / rows * n_samples) / (10 ** (SNR / 10))
    # print(variance)
    n = np.random.randn(rows, 1) * np.sqrt(variance)
    B_n = B + n
    X_hat = somp(A, B_n, variance, k)
    print(X)
    print(X_hat, X_hat.shape)
