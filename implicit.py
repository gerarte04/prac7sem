import numpy as np

def tridiagonal_solve(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    # прямой ход
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] - a[i] * c_prime[i-1]

        if abs(denominator) < 1e-15:
             raise RuntimeError("Division by zero: Thomas method is unstable")

        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator

    # обратный ход

    x = np.zeros(n)
    x[n-1] = d_prime[n-1]

    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

def implicit_step(y, sigma, N):
    B = np.full(N - 1, 1 + 2 * sigma) # главная диагональ

    A = np.full(N - 1, -sigma) # нижняя и верхняя диагонали
    C = np.full(N - 1, -sigma)

    d = y[1:N] # правая часть

    y_internal_next = tridiagonal_solve(A, B, C, d)

    y_new = np.zeros(N + 1)
    y_new[1:N] = y_internal_next
    y_new[0] = 0
    y_new[-1] = 0
    
    return y_new
