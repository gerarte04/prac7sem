import numpy as np

def implicit_step(y_n, A):
    M = A.shape[0]
    
    # a - поддиагональ (индексы 1..M-1)
    a = np.zeros(M)
    a[1:] = np.diag(A, k=-1) 
    
    # b - главная диагональ (индексы 0..M-1)
    b = np.diag(A).copy() 
    
    # c - наддиагональ (индексы 0..M-2)
    c = np.zeros(M)
    c[:-1] = np.diag(A, k=1)
    
    # d - правая часть (внутренние точки с прошлого шага)
    d = y_n[1:-1].copy()

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
    
    return y_n_plus_1
