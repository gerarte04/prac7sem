import numpy as np
from implicit import solve_step_newton

def adaptive_step(problem, u, dt, eps, dt_min, dt_max, safety, max_grow, min_shrink):
    """
    Выполняет один адаптивный шаг по времени.
    Возвращает: u_next, dt_used, dt_next, err, step_count_local
    """
    step_count_local = 0
    
    while True:
        u_full = solve_step_newton(problem, u, dt)
        if u_full is None or np.any(np.isnan(u_full)) or np.any(np.isinf(u_full)):
            dt = max(dt * 0.5, dt_min)
            if dt <= dt_min + 1e-20:
                print("dt упал до минимума и шаг не сходится — остановка")
                raise RuntimeError("Time step too small")
            continue

        # Два полушага
        u_half = solve_step_newton(problem, u, dt/2.0)
        if u_half is None:
            dt = max(dt * 0.5, dt_min)
            if dt <= dt_min + 1e-20:
                print("dt упал до минимума на полушаге — остановка")
                raise RuntimeError("Time step too small")
            continue

        u_two = solve_step_newton(problem, u_half, dt/2.0)
        if u_two is None or np.any(np.isnan(u_two)) or np.any(np.isinf(u_two)):
            dt = max(dt * 0.5, dt_min)
            if dt <= dt_min + 1e-20:
                print("dt упал до минимума на втором полушаге — остановка")
                raise RuntimeError("Time step too small")
            continue

        # L_inf
        err = np.max(np.abs(u_two - u_full))

        if err > eps and dt > dt_min:
            dt_old = dt
            dt = max(dt * 0.4, dt_min)
            step_count_local += 1
            continue

        # Успешный шаг
        step_count_local += 1
        
        if err == 0.0:
            factor = max_grow
        else:
            factor = safety * (eps / err) ** (1.0)  # p = 1
            factor = min(max(factor, min_shrink), max_grow)

        dt_new = dt * factor
        dt_new = min(max(dt_new, dt_min), dt_max)

        return u_two, dt, dt_new, err, step_count_local

