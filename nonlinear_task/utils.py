import numpy as np

def l2_norm(vec):
    return np.sqrt(np.sum(vec**2))

def build_nonlinear_implicit_matrix(y_current, tau, h, k_func, c_func):
    """
    Строит трехдиагональную матрицу для линеаризованной неявной схемы.
    """
    M = len(y_current) - 2 # Количество внутренних точек
    sigma = tau / h**2

    # вычисление аргументов для k
    # a_{M+1} = k((y_M + y_{M+1})/2)
    y_mid = 0.5 * (y_current[:-1] + y_current[1:])
    a_vals = k_func(y_mid)
    
    # Вектор c_i для внутренних точек (индексы 1..M)
    c_vals = c_func(y_current[1:-1])
    
    # Диагонали матрицы размером M x M
    
    # Главная диагональ: c_i + sigma * (a_{i+1} + a_i)
    a_plus = a_vals[1:M+1]
    a_curr = a_vals[0:M]
    main_diag = c_vals + sigma * (a_plus + a_curr)
    
    # Поддиагональ (k=-1): -sigma * a_i
    sub_diag_vals = -sigma * a_vals[1:M]
    
    # Наддиагональ (k=1): -sigma * a_{i+1}
    sup_diag_vals = -sigma * a_vals[1:M]
    
    matrix = np.diag(main_diag) + \
             np.diag(sub_diag_vals, k=-1) + \
             np.diag(sup_diag_vals, k=1)
             
    return matrix, a_vals, c_vals
