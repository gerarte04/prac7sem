import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from adaptive_schemes import adaptive_explicit_step, adaptive_implicit_step
from visualization import (create_animation_figure, 
                           update_animation_plots, 
                           save_final_results)


# ==============================================================================
# ПАРАМЕТРЫ ЗАДАЧИ
# ==============================================================================

L = 1.0       
Nx = 50       
h = L / Nx    
x = np.linspace(0, L, Nx + 1) 

# Параметры временной сетки
sigma = 0.25
tau = sigma * h**2  
T_final = 0.2       

# Параметры адаптивного шага
accuracy = 1e-4           
max_koeff_sigma = 0.5     

M = Nx - 1 

print("="*70)
print("РЕШЕНИЕ КВАЗИЛИНЕЙНОГО УРАВНЕНИЯ ТЕПЛОПРОВОДНОСТИ")
print("c(u) du/dt = d/dx(k(u) du/dx) + f(u)")
print("="*70)

# ==============================================================================
# КОЭФФИЦИЕНТЫ И ФУНКЦИИ
# ==============================================================================

def k_func(u):
    """Коэффициент теплопроводности k(u)"""
    return 1.0 + 1.0 * u**2 

def c_func(u):
    """Теплоемкость c(u)"""
    return 1.0 + 0.1 * u**2

def f_func(u, x, t):
    return 0.0

def u1_boundary(t):
    return 0.0

def u2_boundary(t):
    return 0.0

def initial_condition(x):
    return np.exp(-100 * (x - L/2)**2) * 5.0 # Амплитуда 5, чтобы нелинейность сыграла

print(f"Функции:")
print(f"  k(u) = 1 + u^2")
print(f"  c(u) = 1 + 0.1 * u^2")
print(f"  f(u) = 0")
print(f"  u0(x) = 5 * exp(-100*(x-0.5)^2)")


# ==============================================================================
# ИНИЦИАЛИЗАЦИЯ
# ==============================================================================

y0 = initial_condition(x)
y0[0] = u1_boundary(0)
y0[-1] = u2_boundary(0)

y_exp = y0.copy()
y_imp = y0.copy()

t_history_exp = [0.0]
t_history_imp = [0.0]
tau_history_exp = [tau]
tau_history_imp = [tau]
error_history_exp = [0.0]
error_history_imp = [0.0]

t_exp = 0.0
t_imp = 0.0
tau_exp = tau
tau_imp = tau

step_count_exp = 0
step_count_imp = 0

cur_err_exp = 0.0
cur_err_imp = 0.0

finished_exp = False
finished_imp = False


# ==============================================================================
# АНИМАЦИЯ И РАСЧЕТ
# ==============================================================================

fig, axes_dict, lines_dict, texts_dict = create_animation_figure(
    x, y0, tau, T_final, accuracy
)

def init():
    lines_dict['line_exp'].set_ydata(y0)
    lines_dict['line_imp'].set_ydata(y0)
    texts_dict['text_exp'].set_text('')
    texts_dict['text_imp'].set_text('')
    lines_dict['line_tau_exp'].set_data([], [])
    lines_dict['line_tau_imp'].set_data([], [])
    lines_dict['line_err_exp'].set_data([], [])
    lines_dict['line_err_imp'].set_data([], [])
    return (lines_dict['line_exp'], lines_dict['line_imp'], 
            texts_dict['text_exp'], texts_dict['text_imp'],
            lines_dict['line_tau_exp'], lines_dict['line_tau_imp'], 
            lines_dict['line_err_exp'], lines_dict['line_err_imp'])

def update(frame):
    global y_exp, y_imp, t_exp, t_imp, tau_exp, tau_imp
    global step_count_exp, step_count_imp, finished_exp, finished_imp
    global cur_err_exp, cur_err_imp
    
    # Явная схема
    if not finished_exp and t_exp < T_final:
        if t_exp + tau_exp > T_final:
            tau_exp = T_final - t_exp
        
        y_exp, tau_exp, cur_err_exp, crushed, increased = adaptive_explicit_step(
            y_exp, t_exp, tau_exp, h, max_koeff_sigma, accuracy,
            x, k_func, c_func, f_func, u1_boundary, u2_boundary
        )
        
        t_exp += tau_exp
        step_count_exp += 1
        
        t_history_exp.append(t_exp)
        tau_history_exp.append(tau_exp)
        error_history_exp.append(cur_err_exp)
        
        if step_count_exp % 10 == 0 or crushed:
             print(f"[ЯВН] Шаг {step_count_exp}: t={t_exp:.4f}, tau={tau_exp:.2e}, err={cur_err_exp:.2e}")

        if t_exp >= T_final:
            finished_exp = True
            print(f"\n[ЯВН] Завершено. Шагов: {step_count_exp}")
    
    # Неявная схема
    if not finished_imp and t_imp < T_final:
        if t_imp + tau_imp > T_final:
            tau_imp = T_final - t_imp
        
        y_imp, tau_imp, cur_err_imp, crushed, increased = adaptive_implicit_step(
            y_imp, t_imp, tau_imp, h, accuracy, M,
            x, k_func, c_func, f_func, u1_boundary, u2_boundary
        )
        
        t_imp += tau_imp
        step_count_imp += 1
        
        t_history_imp.append(t_imp)
        tau_history_imp.append(tau_imp)
        error_history_imp.append(cur_err_imp)
        
        if step_count_imp % 10 == 0 or crushed:
             print(f"[НЕЯВ] Шаг {step_count_imp}: t={t_imp:.4f}, tau={tau_imp:.2e}, err={cur_err_imp:.2e}")

        if t_imp >= T_final:
            finished_imp = True
            print(f"\n[НЕЯВ] Завершено. Шагов: {step_count_imp}")
    
    update_animation_plots(
        lines_dict, texts_dict,
        y_exp, y_imp,
        t_exp, t_imp,
        tau_exp, tau_imp,
        cur_err_exp, cur_err_imp,
        step_count_exp, step_count_imp,
        t_history_exp, t_history_imp,
        tau_history_exp, tau_history_imp,
        error_history_exp, error_history_imp,
        finished_exp, finished_imp,
        T_final
    )
    
    if finished_exp and finished_imp:
        ani.event_source.stop()
        save_final_results(
            x, y0, y_exp, y_imp,
            t_history_exp, t_history_imp,
            tau_history_exp, tau_history_imp,
            error_history_exp, error_history_imp,
            T_final, accuracy
        )
    
    return (lines_dict['line_exp'], lines_dict['line_imp'], 
            texts_dict['text_exp'], texts_dict['text_imp'],
            lines_dict['line_tau_exp'], lines_dict['line_tau_imp'], 
            lines_dict['line_err_exp'], lines_dict['line_err_imp'])

ani = FuncAnimation(fig, update, frames=2000, init_func=init, blit=False, interval=1)
plt.show()
