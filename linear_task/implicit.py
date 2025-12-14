import numpy as np

def implicit_step(y_n, A, t, tau, h, x, f, u1, u2):
    M = A.shape[0]
    sigma = tau / h**2
    
    # a - поддиагональ (индексы 1..M-1)
    a = np.zeros(M)
    a[1:] = np.diag(A, k=-1) 
    
    # b - главная диагональ (индексы 0..M-1)
    b = np.diag(A).copy() 
    
    # c - наддиагональ (индексы 0..M-2)
    c = np.zeros(M)
    c[:-1] = np.diag(A, k=1)
    
    # d - правая часть
    # (y_new - y_old)/tau = Delta y_new + f(x, t_new)
    # -sigma*y_{i-1}^{n+1} + (1+2sigma)*y_i^{n+1} - sigma*y_{i+1}^{n+1} = y_i^n + tau*f(x_i, t_{n+1})
    
    d = y_n[1:-1].copy() + tau * f(x[1:-1], t + tau)

    # Учет граничных условий в правой части
    # Для i=1: ... - sigma*y_0^{n+1} = ... -> добавляем sigma*u1(t+tau) к правой части
    # Для i=M: ... - sigma*y_{M+1}^{n+1} = ... -> добавляем sigma*u2(t+tau) к правой части
    
    d[0] += sigma * u1(t + tau)
    d[-1] += sigma * u2(t + tau)

    # Прямой ход

    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]

    for i in range(1, M):
        temp = b[i] - a[i] * c[i-1]
        if abs(temp) < 1e-15:
            raise np.linalg.LinAlgError("Метод прогонки неустойчив: деление на ноль.")
            
        c[i] = c[i] / temp
        d[i] = (d[i] - a[i] * d[i-1]) / temp

    # Обратный ход

    y_sol = np.zeros(M)
    y_sol[-1] = d[-1]
    
    for i in range(M - 2, -1, -1):
        y_sol[i] = d[i] - c[i] * y_sol[i+1]

    y_n_plus_1 = np.zeros_like(y_n)
    y_n_plus_1[1:-1] = y_sol
    
    # Граничные условия
    y_n_plus_1[0] = u1(t + tau)
    y_n_plus_1[-1] = u2(t + tau)
    
    return y_n_plus_1
