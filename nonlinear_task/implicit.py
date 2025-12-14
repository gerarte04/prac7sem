import numpy as np
from utils import build_nonlinear_implicit_matrix

def implicit_step(y_n, t, tau, h, x, k_func, c_func, f_func, u1, u2):
    sigma = tau / h**2
    M = len(y_n) - 2
    
    # Строим матрицу A (зависит от y_n)
    # также строим массивы коэффициентов a и c для построения правой части
    A, a_vals, c_vals = build_nonlinear_implicit_matrix(y_n, tau, h, k_func, c_func)
    
    # Правая часть d
    # ... = c_i * y_i^n + tau * f_i^n
    rhs = c_vals * y_n[1:-1] + tau * f_func(y_n[1:-1], x[1:-1], t)
    
    # Граничные значения на новом слое
    val_u1 = u1(t + tau)
    val_u2 = u2(t + tau)

    # Для первого уравнения (i=1, row=0):
    # -sigma * a_1 * y_0^{n+1} переносим вправо -> + sigma * a_1 * u1
    rhs[0] += sigma * a_vals[0] * val_u1
    
    # Для последнего уравнения (i=M, row=M-1):
    # -sigma * a_{M+1} * y_{M+1}^{n+1} переносим вправо -> + sigma * a_{M+1} * u2
    rhs[-1] += sigma * a_vals[M] * val_u2
    
    # решаем томасом
    y_sol = solve_tridiagonal(A, rhs)
    
    y_n_plus_1 = np.zeros_like(y_n)
    y_n_plus_1[1:-1] = y_sol
    y_n_plus_1[0] = val_u1
    y_n_plus_1[-1] = val_u2
    
    return y_n_plus_1


def solve_tridiagonal(A, d):
    """
    Метод прогонки для трехдиагональной матрицы Ax = d.
    """
    M = len(d)
    
    # a - поддиагональ (ниже главной)
    a = np.zeros(M)
    a[1:] = np.diag(A, k=-1)
    
    # b - главная диагональ
    b = np.diag(A).copy()
    
    # c - наддиагональ (выше главной)
    c = np.zeros(M)
    c[:-1] = np.diag(A, k=1)
    
    # Прямой ход
    # c_i = c_i / (b_i - a_i * c_{i-1})

    d_vec = d.copy()
    c_vec = c.copy()
    
    c_vec[0] = c_vec[0] / b[0]
    d_vec[0] = d_vec[0] / b[0]
    
    for i in range(1, M):
        temp = b[i] - a[i] * c_vec[i-1]
        if abs(temp) < 1e-15:
             # если не получилось томасом решаем по другому
             return np.linalg.solve(A, d)
        
        c_vec[i] = c_vec[i] / temp
        d_vec[i] = (d_vec[i] - a[i] * d_vec[i-1]) / temp
        
    # Обратный ход
    x = np.zeros(M)
    x[-1] = d_vec[-1]
    
    for i in range(M - 2, -1, -1):
        x[i] = d_vec[i] - c_vec[i] * x[i+1]
        
    return x
