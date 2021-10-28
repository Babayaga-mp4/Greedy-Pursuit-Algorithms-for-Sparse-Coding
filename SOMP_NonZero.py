import numpy as np
from numpy import linalg as LA
from scipy.sparse import rand
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt

def somp(A, b, variance, *args):
    r = b
    samples = b.shape[1]
    cols = A.shape[1]
    rows = A.shape[0]
    if args == (): k = cols / 2
    else: k = args[0]
    A_reduced = np.empty((rows, 0))
    X = np.zeros((cols, samples))
    s = []
    error_power = float('inf')
    while (error_power >= 1.15 * variance) or (k > 0):
        Arr = np.sum(abs(np.matmul(A.T, r)), axis= 1)                          # Sum along the rows
        next_col = np.where(Arr == np.amax(Arr))[0]
        s.append(next_col.tolist()[0])
        A_reduced = np.column_stack((A_reduced, A[:, next_col]))
        Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
        X[s, :] = Xs
        b_hat = np.matmul(A, X)
        r = b - b_hat
        error_power = (LA.norm(r) ** 2 )/ (rows * samples)
        k -= 1
    return X


if __name__ == '__main__':
    rows = 200
    cols = 500
    # k = 3
    A = np.random.randn(rows, cols)
    for i in range(cols):
        A[:, i] = A[:, i] / LA.norm(A[:, i])

    n_samples, abs_error = 5, []
    hamming_distance = []

    for counter in range(1, 30):
        error, h_dist = 0, 0
        k = counter

        for cdx in range(1000):

            X = np.zeros((cols, n_samples))
            # dense = np.random.randn(k, n_samples)
            # print(dense)
            track = []
            idx = 0

            while idx < k:
                rand_ind = random.randint(0, cols - 1)
                if rand_ind not in track:
                    X[rand_ind, :] = np.random.rand(1, n_samples)  # Generate a random row vector
                    idx += 1
                    track.append(rand_ind)

            # print(X)

            B = np.matmul(A, X)
            SNR = 50
            variance = (LA.norm(B) ** 2 / rows * n_samples) / (10 ** (SNR / 10))
            n = np.random.randn(rows, 1) * np.sqrt(variance)
            B_n = B + n
            X_hat = somp(A, B_n, variance, k)
            # print(X)
            # print(X_hat, X_hat.shape)
            B_hat = np.matmul(A, X_hat)
            # X_hat_ham, X_ham = np.zeros((cols, 1)), np.zeros((cols, 1))
            # for kdx in range(cols):
            #     if X_hat[:,0][kdx] != 0:
            #         X_hat_ham[kdx] = 1
            #     elif X[:,0][kdx] != 0:
            #         X_ham[kdx] = 1

            # h_dist += distance.hamming(X_ham, X_hat_ham)/n_samples

            error += ((LA.norm(B - B_hat)) ** 2) / (n_samples * 1000 * rows)
        abs_error.append(error)
        # hamming_distance.append(h_dist/1000)
    plt.plot([idx for idx in range(1, 30)], abs_error)
    # plt.plot([idx for idx in range(1, 30)], hamming_distance)
    plt.show()



