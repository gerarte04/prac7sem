import numpy as np

def build_implicit_matrix(sigma, M):
    return np.diag((1 + 2 * sigma) * np.ones(M)) + \
           np.diag(-sigma * np.ones(M - 1), k=1) + \
           np.diag(-sigma * np.ones(M - 1), k=-1)


def l2_norm(vec):
    return np.sqrt(np.sum(vec**2))


def lc_norm(vec):
    return np.max(np.abs(vec))
