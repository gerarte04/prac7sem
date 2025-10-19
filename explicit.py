import numpy as np

def explicit_step(y_n, sigma):
    y_n_plus_1 = np.zeros_like(y_n)
    y_n_plus_1[1:-1] = y_n[1:-1] + sigma * (y_n[0:-2] - 2 * y_n[1:-1] + y_n[2:])

    return y_n_plus_1
