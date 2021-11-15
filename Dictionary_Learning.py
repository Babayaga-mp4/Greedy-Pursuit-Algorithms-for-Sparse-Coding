import numpy as np
from numpy import linalg as LA
from scipy.sparse import rand
import matplotlib.pyplot as plt

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
        A_reduced = np.column_stack((A_reduced, (A[:, next_col[0]])))
        Xs = np.matmul((np.linalg.pinv(A_reduced)), b)
        X[s] = Xs
        b_hat = np.matmul(A, X)
        r_prev = b - b_hat
        i += 1
    return X


def learn_dict(rows, cols, non_zeros, b, *args):
    samples = b.shape[1]
    num_iter = 20

    if args != ():
        input_variance = args[0]
        c = 1.15

    # Randomly Initializing a dictionary

    dictionary = np.random.randn(rows, cols)
    for idx in range(cols):
        dictionary[:, idx] = dictionary[:, idx] / LA.norm(dictionary[:, idx])  # (20, 30)

    iter_counter = 0
    error = float('inf')  # Make Infinity

    while iter_counter < num_iter:

    # while error > c*input_variance:

        # Sparse Coding

        X_hat = np.empty((cols, 0))
        for idx in range(samples):
            new_X_hat = omp(non_zeros, dictionary, b[:, idx])
            X_hat = np.concatenate((X_hat, new_X_hat), axis=1)  # (30, 10)

        # Dictionary Update

        for kdx in range(cols - 1):

            w_k = [idx for idx in range(samples) if X_hat[kdx, idx] != 0]

            if len(w_k) == 0:
                continue

            # X_hat_k = X_hat[kdx:kdx+1, :].T[w_k]
            # b_k = b[w_k]
            dictionary[:, kdx:kdx + 1] = 0
            E = b - np.matmul(dictionary, X_hat)  # (20,10)       # Check the coherence between E and E_k
            E_k = E.T[w_k].T  # (20, |w_k|)

            U, S, V = LA.svd(E_k)
            dictionary[:, kdx:kdx + 1] = U[:, 0:1]

            X_hat_k = V[:, 0:1] * S[0]
            X_hat[kdx:kdx + 1, :].T[w_k] = X_hat_k

            residual = b - np.matmul(dictionary, X_hat)
            error = (LA.norm(residual) ** 2) / samples

        iter_counter += 1
    # print(iter_counter)
    if error < input_variance: print('Yes', error, input_variance)
    return dictionary


if __name__ == '__main__':
    rows = 300
    cols = 600
    non_zeros = 4
    n_samples = 100
    # Dictionary
    # SNR = 50
    fina = []
    for SNR in range(10, 30, 5):
        print(SNR)
        recovered = []
        counter = 0
        for idx in range(10):

            A = np.random.randn(rows, cols)
            for i in range(cols):
                A[:, i] = A[:, i] / LA.norm(A[:, i])
            # print(A)



            # Expected Sparse Signals
            X = rand(cols, n_samples,
                    density=non_zeros / cols).todense()

            # Y
            b = A * X
            # SNR = 10
            variance = (LA.norm(b) ** 2 / rows * n_samples) / (10 ** (SNR / 10))
            n = np.random.randn(rows, 1) * np.sqrt(variance)
            b_n = b + n

            # Testing

            learned_dict = learn_dict(rows, cols, non_zeros, b_n, variance)

            for mdx in range(cols):
              d = A[:, mdx]
              for ndx in range(cols):
                dt = learned_dict[:, ndx]
                s = abs(np.matmul(d.T, dt))
                # print(s)
                if s >= 0.9:
                  counter += 1
                  break
            # print('counter:', counter/10)
            # print('variance:', variance)
            residual = b_n - np.matmul(A, X)
            error = (LA.norm(residual) ** 2)/(rows*n_samples)
            recovered.append(counter/10)
        print(recovered)
        fina.append(counter/10)
        print(fina)
        # print(sum(recovered))
    plt.plot([idx for idx in range(10, 30, 5)], fina)
    plt.show()