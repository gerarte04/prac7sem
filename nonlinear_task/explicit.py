import numpy as np

def explicit_step(y_n, t, tau, h, x, k_func, c_func, f_func, u1, u2):
    sigma = tau / h**2
    y_n_plus_1 = np.zeros_like(y_n)
    
    # Коэффициенты
    y_mid = 0.5 * (y_n[:-1] + y_n[1:])
    a_vals = k_func(y_mid) 
    # a_vals[i] -> a_{i+1}
    # a_vals[i-1] -> a_i
    
    c_vals = c_func(y_n[1:-1])
    
    # Разностный оператор теплопроводности Lambda y
    # term1 = a_{i+1} * (y_{i+1} - y_i)
    term1 = a_vals[1:] * (y_n[2:] - y_n[1:-1])
    
    # term2 = a_i * (y_i - y_{i-1})
    term2 = a_vals[:-1] * (y_n[1:-1] - y_n[:-2])
    
    diff_op = sigma * (term1 - term2)
    
    # Источник
    source = tau * f_func(y_n[1:-1], x[1:-1], t)
    
    # Обновление внутренних точек
    # y^{n+1} = y^n + (diff_op + source) / c_vals
    y_n_plus_1[1:-1] = y_n[1:-1] + (diff_op + source) / c_vals
    
    # Граничные условия
    y_n_plus_1[0] = u1(t + tau)
    y_n_plus_1[-1] = u2(t + tau)
    
    return y_n_plus_1
