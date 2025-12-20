import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def compute_residual(problem, u, u_prev, dt):
    N = len(u)
    h = problem.h
    alpha = dt / (h * h)
    
    F = np.zeros(N)
    
    # Векторизованные вычисления для внутренних узлов
    u_center = u[1:-1]
    u_right = u[2:]
    u_left = u[:-2]
    
    u_plus = 0.5 * (u_center + u_right)
    u_minus = 0.5 * (u_center + u_left)
    
    k_plus = problem.k_func(u_plus)
    k_minus = problem.k_func(u_minus)
    
    flux_plus = k_plus * (u_right - u_center)
    flux_minus = k_minus * (u_center - u_left)
    
    F[1:-1] = u_prev[1:-1] + dt * problem.f_func(u_center) + alpha * (flux_plus - flux_minus) - u_center
    
    # Неймана
    F[0] = u[0] - u[1]
    F[-1] = u[-1] - u[-2]
    
    return F

def build_jacobian(problem, u, u_prev, dt):
    N = len(u)
    h = problem.h
    alpha = dt / (h * h)

    main_diag = np.zeros(N)
    lower_diag = np.zeros(N-1)
    upper_diag = np.zeros(N-1)

    # Векторизованные вычисления
    u_center = u[1:-1]
    u_right = u[2:]
    u_left = u[:-2]

    u_plus = 0.5 * (u_center + u_right)
    u_minus = 0.5 * (u_center + u_left)

    k_plus = problem.k_func(u_plus)
    k_minus = problem.k_func(u_minus)
    dk_du_plus = problem.dk_du_func(u_plus)
    dk_du_minus = problem.dk_du_func(u_minus)

    # lower_diag[i-1] где i от 1 до N-2 -> индексы 0 до N-3
    lower_diag[:-1] = -alpha * (0.5 * dk_du_minus * (u_center - u_left) - k_minus)
    
    # upper_diag[i] где i от 1 до N-2 -> индексы 1 до N-2
    upper_diag[1:] = alpha * (0.5 * dk_du_plus * (u_right - u_center) + k_plus)

    flux_minus_dui = 0.5 * dk_du_minus * (u_center - u_left) + k_minus
    flux_plus_dui = 0.5 * dk_du_plus * (u_right - u_center) - k_plus

    main_diag[1:-1] = dt * problem.df_du_func(u_center) - 1.0 + alpha * (flux_plus_dui - flux_minus_dui)


    main_diag[0] = 1.0
    upper_diag[0] = -1.0

    main_diag[-1] = 1.0
    lower_diag[-1] = -1.0

    J = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
    return J

def solve_step_newton(problem, u_prev, dt):
    u = u_prev.copy()
    
    for newton_iter in range(problem.max_newton_iter):
        F = compute_residual(problem, u, u_prev, dt)
        residual_norm = np.linalg.norm(F, np.inf)
        
        if residual_norm < problem.newton_tol:
            return u  # сошлось
        
        J = build_jacobian(problem, u, u_prev, dt)
        d = -F
        
        try:
            # Использование spsolve из scipy вместо самописной прогонки
            delta_u = spsolve(J, d)
        except Exception as e:
            return None
        
        max_delta = np.max(np.abs(delta_u))
        if max_delta > 10.0:
            delta_u = delta_u * (10.0 / max_delta)
        
        u_new = u + delta_u
        u_new = np.maximum(u_new, 0.0)
        u = u_new
    
    F_final = compute_residual(problem, u, u_prev, dt)
    if np.linalg.norm(F_final, np.inf) < 1e-6:
        return u
    else:
        return None
