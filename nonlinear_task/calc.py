import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from adaptive_schemes import adaptive_step
from visualization import (create_animation_figure, 
                           update_animation_plots, 
                           save_final_results)

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


# Глобальные переменные для анимации
problem = None
u = None
t = 0.0
dt = 0.0
step_count = 0
t_history = []
dt_history = []
error_history = []
finished = False
cur_err = 0.0
eps = 1e-6
dt_min = 1e-13
dt_max = 0.1
save_every = 10
safety = 0.9
max_grow = 2.0
min_shrink = 0.2


def init_animation():
    """Инициализация анимации"""
    # Начальная линия уже задана при создании
    return (lines_dict['line_num'], lines_dict['line_exact'], 
            lines_dict['line_err_prof'],
            lines_dict['line_dt'], lines_dict['line_err_hist'], 
            texts_dict['text_info'])

def update(frame):
    """Обновление кадра анимации"""
    global u, t, dt, step_count, finished, cur_err
    
    if not finished and t < problem.T:
        if t + dt > problem.T:
            dt = problem.T - t

        try:
            u_two, dt_used, dt_new, err, local_steps = adaptive_step(
                problem, u, dt, eps, dt_min, dt_max, safety, max_grow, min_shrink
            )
            
            u = u_two.copy()
            t += dt_used
            dt = dt_new # Шаг на следующий раз
            cur_err = err
            step_count += local_steps
            
            t_history.append(t)
            dt_history.append(dt_used)
            error_history.append(err)

            if dt <= dt_min + 1e-20 and t < problem.T:
                 print("dt достиг минимума при адаптации — прекращаем интегрирование")
                 finished = True

            if t >= problem.T - 1e-15:
                finished = True
                print(f"Достигнуто конечное время t={t:.6f}")

            if step_count % 10 == 0:
                 print(f"Шаг {step_count}: t={t:.6f}, dt={dt_used:.3e}, max(u)={np.max(u):.6e}")
                 
        except RuntimeError:
            print("dt достиг минимума при адаптации — прекращаем интегрирование")
            finished = True

    # Обновление графиков
    u_exact_val = problem.u_exact(problem.x, t)
    
    update_animation_plots(
        lines_dict, texts_dict,
        u, u_exact_val,
        t, dt,
        cur_err,
        step_count,
        t_history,
        dt_history,
        error_history,
        finished,
        problem.T
    )

    if finished and frame > 0 and frame % 50 == 0: # Stop if finished, but keep checking occasionally or just let it spin
         # ani.event_source.stop() # Can stop here if desired
         pass

    return (lines_dict['line_num'], lines_dict['line_exact'], 
            lines_dict['line_err_prof'],
            lines_dict['line_dt'], lines_dict['line_err_hist'], 
            texts_dict['text_info'])


def main():
    global problem, u, t, dt, lines_dict, texts_dict, t_history, dt_history, error_history, finished
    
    problem = ThermalLocalizationProblem()
    
    # Параметры расчета
    dt0 = 1e-3
    t = 0.0
    u = problem.u0_func(problem.x)
    dt = dt0
    
    # Инициализация истории
    t_history = [t]
    dt_history = [dt] # Начальный шаг
    error_history = [0.0]

    print(f"Начальное max(u) = {np.max(u):.6e}")

    # Создание фигуры
    fig, axes_dict, lines_dict, texts_dict_loc = create_animation_figure(
        problem.x, u, dt, problem.T, eps
    )
    texts_dict = texts_dict_loc

    # Запуск анимации
    max_frames = 3000
    ani = FuncAnimation(fig, update, frames=max_frames,
                        init_func=init_animation, blit=False, interval=1, repeat=False)
    
    plt.show()

    # Финальные результаты
    if len(t_history) > 0:
        u_final = u
        u_exact_final = problem.u_exact(problem.x, t_history[-1])
        error = u_final - u_exact_final
        error_linf = np.max(np.abs(error))
        print(f"Финальный момент: t = {t_history[-1]:.6f}")
        print(f"max числ = {np.max(u_final):.6e}, max точн = {np.max(u_exact_final):.6e}")
        print(f"Ошибка Linf = {error_linf:.3e}")
        
        save_final_results(
            problem.x, u_final, u_exact_final,
            t_history, dt_history, error_history,
            problem.T, eps
        )
    else:
        print("Нет сохранённых решений.")

if __name__ == "__main__":
    main()
