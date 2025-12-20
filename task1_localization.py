import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags


class ThermalLocalizationProblem:
    def __init__(self):
        self.name = "Localized nonlinear heating"
        # Параметры задачи
        self.L = 10.0
        self.T = 0.98  # Целевое время
        self.Nx = 250
        self.h = self.L / self.Nx
        self.x = np.linspace(0, self.L, self.Nx + 1)
        self.xc = 5.0
        self.R = np.pi * np.sqrt(2.0)
        self.dt = 0.01  

        
        # Параметры метода Ньютона
        self.newton_tol = 1e-12
        self.max_newton_iter = 40
    
    def k_func(self, u):
        # return np.maximum(u, 1e-12)
        return u
    
    def dk_du_func(self, u):
        return np.ones_like(u)
    
    def f_func(self, u):
        return u * u
    
    def df_du_func(self, u):
        return 2.0 * u
    
    def u0_func(self, x):
        y = x - self.xc
        result = np.zeros_like(x)
        mask = np.abs(y) <= self.R
        arg = np.pi * y[mask] / (2.0 * self.R)
        result[mask] = (4.0 / 3.0) * np.cos(arg) * np.cos(arg)
        return result
    
    def u_exact(self, x, t):

        if t >= 1.0:
            return np.zeros_like(x)
        y = x - self.xc
        result = np.zeros_like(x)
        mask = np.abs(y) <= self.R
        arg = np.pi * y[mask] / (2.0 * self.R)
        u0_vals = (4.0 / 3.0) * np.cos(arg) * np.cos(arg)
        result[mask] = u0_vals / (1.0 - t)
        return result


def compute_residual(problem, u, u_prev, dt):
    N = len(u)
    h = problem.h
    alpha = dt / (h * h)
    
    F = np.zeros(N)
    

    for i in range(1, N-1):
        u_plus = 0.5 * (u[i] + u[i+1])
        u_minus = 0.5 * (u[i] + u[i-1])
        
        k_plus = problem.k_func(u_plus)
        k_minus = problem.k_func(u_minus)
        
        flux_plus = k_plus * (u[i+1] - u[i])
        flux_minus = k_minus * (u[i] - u[i-1])
        
        F[i] = u_prev[i] + dt * problem.f_func(u[i]) + alpha * (flux_plus - flux_minus) - u[i]
    
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

    for i in range(1, N-1):
        u_i = u[i]
        u_ip1 = u[i+1]
        u_im1 = u[i-1]

        u_plus = 0.5 * (u_i + u_ip1)
        u_minus = 0.5 * (u_i + u_im1)

        k_plus = problem.k_func(u_plus)
        k_minus = problem.k_func(u_minus)
        dk_du_plus = problem.dk_du_func(u_plus)
        dk_du_minus = problem.dk_du_func(u_minus)

        lower_diag[i-1] = -alpha * (0.5 * dk_du_minus * (u_i - u_im1) - k_minus)
        upper_diag[i] = alpha * (0.5 * dk_du_plus * (u_ip1 - u_i) + k_plus)

        flux_minus_dui = 0.5 * dk_du_minus * (u_i - u_im1) + k_minus
        flux_plus_dui = 0.5 * dk_du_plus * (u_ip1 - u_i) - k_plus

        main_diag[i] = dt * problem.df_du_func(u_i) - 1.0 + alpha * (flux_plus_dui - flux_minus_dui)


    main_diag[0] = 1.0
    upper_diag[0] = -1.0

    main_diag[-1] = 1.0
    lower_diag[-1] = -1.0

    J = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
    return J

def tridiagonal_solve(a, b, c, d):

    n = len(b)
    cp = np.zeros(n-1)
    dp = np.zeros(n)
    x = np.zeros(n)
    
    # прямой ход
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])
    
    # обратный ход
    x[n-1] = dp[n-1]
    for i in reversed(range(n-1)):
        x[i] = dp[i] - cp[i] * x[i+1]
    
    return x


def solve_step_newton(problem, u_prev, dt):

    u = u_prev.copy()
    
    for newton_iter in range(problem.max_newton_iter):
        F = compute_residual(problem, u, u_prev, dt)
        residual_norm = np.linalg.norm(F, np.inf)
        
        if residual_norm < problem.newton_tol:
            return u  # сошлось
        
        J = build_jacobian(problem, u, u_prev, dt)
        

        lower_diag = J.diagonal(-1)
        main_diag = J.diagonal(0)
        upper_diag = J.diagonal(1)
        d = -F
        
        try:
            delta_u = tridiagonal_solve(lower_diag, main_diag, upper_diag, d)
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



def solve_problem_adaptive(problem, dt_init=None, eps=1e-7, dt_min=1e-20, dt_max=None, 
                           save_every=10, safety=0.9, max_grow=2.0, min_shrink=0.2):
   
    if dt_init is None:
        dt = problem.dt
    else:
        dt = dt_init
    if dt_max is None:
        dt_max = problem.T  

    t = 0.0
    u = problem.u0_func(problem.x)
    solutions = [u.copy()]
    times = [t]

    step_count = 0
    accepted_steps = 0

    print(f"Начальное max(u) = {np.max(u):.6e}, стартовый dt = {dt:.3e}, eps = {eps:.1e}")

    while t < problem.T:
        if t + dt > problem.T:
            dt = problem.T - t

        u_full = solve_step_newton(problem, u, dt)
        if u_full is None or np.any(np.isnan(u_full)) or np.any(np.isinf(u_full)):

            dt = max(dt * 0.5, dt_min)
            if dt <= dt_min + 1e-20:
                print("dt упал до минимума и шаг не сходится — остановка")
                break
            continue

        # Два полушага
        u_half = solve_step_newton(problem, u, dt/2.0)
        if u_half is None:
            dt = max(dt * 0.5, dt_min)
            if dt <= dt_min + 1e-20:
                print("dt упал до минимума на полушаге — остановка")
                break
            continue

        u_two = solve_step_newton(problem, u_half, dt/2.0)
        if u_two is None or np.any(np.isnan(u_two)) or np.any(np.isinf(u_two)):
            dt = max(dt * 0.5, dt_min)
            if dt <= dt_min + 1e-20:
                print("dt упал до минимума на втором полушаге — остановка")
                break
            continue

        # L_inf
        err = np.max(np.abs(u_two - u_full))

        if err > eps and dt > dt_min:
            dt_old = dt
            dt = max(dt * 0.4, dt_min)
            step_count += 1
            continue

        t += dt
        u = u_two.copy()
        accepted_steps += 1
        step_count += 1

        if accepted_steps % save_every == 0 or t >= problem.T - 1e-15:
            solutions.append(u.copy())
            times.append(t)
            max_u = np.max(u)
            exact_max = np.max(problem.u_exact(problem.x, t))
            print(f"t={t:.6f}, accepted_steps={accepted_steps}, dt={dt:.3e}, max(u)={max_u:.6e}, exact_max={exact_max:.6e}")

        if err == 0.0:
            factor = max_grow
        else:
            factor = safety * (eps / err) ** (1.0)  # p = 1

            factor = min(max(factor, min_shrink), max_grow)

        dt_new = dt * factor

        dt = min(max(dt_new, dt_min), dt_max)


        if dt <= dt_min + 1e-20:
            print("dt достиг минимума при адаптации — прекращаем интегрирование")
            break

    print(f"шагов: {step_count}, финальное t = {t:.6f}")
    return np.array(times), np.array(solutions)


def main():
    
    problem = ThermalLocalizationProblem()
    dt0 = 1e-3
    eps = 1e-6
    dt_min = 1e-13
    dt_max = 0.1
    save_every = 10

    times, solutions = solve_problem_adaptive(problem, dt_init=dt0, eps=eps,
                                              dt_min=dt_min, dt_max=dt_max,
                                              save_every=save_every)

    if len(solutions) > 0:
        u_final = solutions[-1]
        u_exact_final = problem.u_exact(problem.x, times[-1])
        error = u_final - u_exact_final
        error_linf = np.max(np.abs(error))
        print(f"Финальный момент: t = {times[-1]:.6f}")
        print(f"max числ = {np.max(u_final):.6e}, max точн = {np.max(u_exact_final):.6e}")
        print(f"Ошибка Linf = {error_linf:.3e}")
    else:
        print("Нет сохранённых решений.")
    
    return problem, times, solutions  


if __name__ == "__main__":
    problem, times, solutions = main()  

    # График изменения шага dt(t)
    def plot_dt(times):
        dt_values = np.diff(times)
        plt.figure(figsize=(8, 4))
        plt.plot(times[:-1], dt_values, lw=2)
        plt.xlabel("t")
        plt.ylabel("dt")
        plt.title("Адаптивный шаг dt(t)")
        plt.grid()
        plt.show()
    plot_dt(times)

    # Эволюция профилей решения u(x,t) 
    def plot_profiles(problem, times, solutions, num=6):
        if len(solutions) == 0:
            print("Нет данных для графика профилей.")
            return

        idx = np.linspace(0, len(times)-1, min(num, len(times))).astype(int)
        plt.figure(figsize=(8, 5))

        colors = plt.cm.viridis(np.linspace(0, 1, len(idx)))

        for c, i in zip(colors, idx):
            u_num = solutions[i]
            u_exact = problem.u_exact(problem.x, times[i])
            plt.plot(problem.x, u_num, color=c, lw=2)
            plt.plot(problem.x, u_exact, '--', color=c, lw=1.5)

        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title("Эволюция профилей: числ. vs точн.")
        plt.grid()
        plt.tight_layout()
        
        plt.plot([], [], color='black', lw=2, label="числ.")
        plt.plot([], [], color='black', lw=1.5, linestyle='--', label="точн.")
        plt.legend()
        plt.show()
    plot_profiles(problem, times, solutions)

    # Ошибка относительно точного решения
    def plot_error(problem, times, solutions):
        t_final = times[-1]
        u_num = solutions[-1]
        u_exact = problem.u_exact(problem.x, t_final)
        error = np.abs(u_num - u_exact)
        plt.figure(figsize=(8, 4))
        plt.plot(problem.x, error, lw=2)
        plt.xlabel("x")
        plt.ylabel("|error|")
        plt.title(f"Ошибка |u - u_exact| при t={t_final:.4f}")
        plt.grid()
        plt.show()
    plot_error(problem, times, solutions)
