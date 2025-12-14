import numpy as np

def explicit_step(y_n, t, tau, h, x, f, u1, u2):
    sigma = tau / h**2
    y_n_plus_1 = np.zeros_like(y_n)
    
    # Явная схема: (y_new - y_old)/tau = (y_{i-1} - 2y_i + y_{i+1})/h^2 + f(x, t)
    # y_new = y_old + sigma*(y_{i-1} - 2y_i + y_{i+1}) + tau*f(x, t)
    
    y_n_plus_1[1:-1] = y_n[1:-1] + sigma * (y_n[0:-2] - 2 * y_n[1:-1] + y_n[2:]) + tau * f(x[1:-1], t)

    # Граничные условия
    y_n_plus_1[0] = u1(t + tau)
    y_n_plus_1[-1] = u2(t + tau)

    return y_n_plus_1
