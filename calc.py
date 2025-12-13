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

L = 1.0       # Длина области
Nx = 50       # Количество *интервалов* по x (точек будет Nx + 1)
h = L / Nx    # Шаг по x
x = np.linspace(0, L, Nx + 1) # Массив узлов по x

# Параметры временной сетки
sigma = 0.25
tau = sigma * h**2  # Начальный шаг по времени
T_final = 0.1       # Конечное время

# Параметры адаптивного шага
accuracy = 1e-4           # Требуемая точность для контроля ошибки
max_koeff_sigma = 0.5     # Максимальное значение sigma для явной схемы

M = Nx - 1  # Размерность СЛАУ (количество внутренних точек)

print("="*70)
print("РЕШЕНИЕ УРАВНЕНИЯ ТЕПЛОПРОВОДНОСТИ")
print("Сравнение явной и неявной схем с адаптивным шагом")
print("="*70)
print(f"\nПараметры задачи:")
print(f"  Область: [0, {L}]")
print(f"  Конечное время: {T_final}")
print(f"\nПараметры сетки:")
print(f"  Nx = {Nx}, h = {h:.4f}")
print(f"  Начальный tau = {tau:.6f}, sigma = {sigma:.3f}")
print(f"\nПараметры адаптации:")
print(f"  Требуемая точность = {accuracy}")
print(f"  Max sigma (явная) = {max_koeff_sigma}")


# ==============================================================================
# НАЧАЛЬНОЕ УСЛОВИЕ
# ==============================================================================

y0 = np.exp(-100 * (x - L/2)**2)
y0[0] = 0
y0[-1] = 0

y_exp = y0.copy()
y_imp = y0.copy()


# ==============================================================================
# ИНИЦИАЛИЗАЦИЯ ДЛЯ АНИМАЦИИ
# ==============================================================================

# История для визуализации
t_history_exp = [0.0]
t_history_imp = [0.0]
tau_history_exp = [tau]
tau_history_imp = [tau]
error_history_exp = [0.0]
error_history_imp = [0.0]

# Текущее время и шаг
t_exp = 0.0
t_imp = 0.0
tau_exp = tau
tau_imp = tau

step_count_exp = 0
step_count_imp = 0

# Текущие ошибки
cur_err_exp = 0.0
cur_err_imp = 0.0

# Флаги завершения
finished_exp = False
finished_imp = False

print("\n" + "="*70)
print("Начало вычислений с адаптивным шагом (анимация)")
print("="*70)


# ==============================================================================
# СОЗДАНИЕ ФИГУРЫ ДЛЯ АНИМАЦИИ
# ==============================================================================

fig, axes_dict, lines_dict, texts_dict = create_animation_figure(
    x, y0, tau, T_final, accuracy
)


# ==============================================================================
# ФУНКЦИИ АНИМАЦИИ
# ==============================================================================

def init():
    """Инициализация анимации"""
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
    """Обновление кадра анимации"""
    global y_exp, y_imp, t_exp, t_imp, tau_exp, tau_imp
    global step_count_exp, step_count_imp, finished_exp, finished_imp
    global cur_err_exp, cur_err_imp
    
    # Обновление явной схемы
    if not finished_exp and t_exp < T_final:
        # Проверка на выход за границу по времени
        if t_exp + tau_exp > T_final:
            tau_exp = T_final - t_exp
        
        y_exp, tau_exp, cur_err_exp, crushed, increased = adaptive_explicit_step(
            y_exp, tau_exp, h, max_koeff_sigma, accuracy
        )
        
        t_exp += tau_exp
        step_count_exp += 1
        
        t_history_exp.append(t_exp)
        tau_history_exp.append(tau_exp)
        error_history_exp.append(cur_err_exp)
        
        if step_count_exp % 10 == 0 or crushed or increased:
            status = ""
            if crushed and increased:
                status = " [CRUSH & INCREASE]"
            elif crushed:
                status = " [CRUSHED]"
            elif increased:
                status = " [INCREASED]"
            print(f"[ЯВН] Шаг {step_count_exp:4d}: t = {t_exp:.6f}, "
                  f"tau = {tau_exp:.6e}, err = {cur_err_exp:.2e}{status}")
        
        if t_exp >= T_final:
            finished_exp = True
            print(f"\n[ЯВН] Явная схема завершена: {step_count_exp} шагов\n")
    
    # Обновление неявной схемы
    if not finished_imp and t_imp < T_final:
        # Проверка на выход за границу по времени
        if t_imp + tau_imp > T_final:
            tau_imp = T_final - t_imp
        
        y_imp, tau_imp, cur_err_imp, crushed, increased = adaptive_implicit_step(
            y_imp, tau_imp, h, accuracy, M
        )
        
        t_imp += tau_imp
        step_count_imp += 1
        
        t_history_imp.append(t_imp)
        tau_history_imp.append(tau_imp)
        error_history_imp.append(cur_err_imp)
        
        if step_count_imp % 10 == 0 or crushed or increased:
            status = ""
            if crushed and increased:
                status = " [CRUSH & INCREASE]"
            elif crushed:
                status = " [CRUSHED]"
            elif increased:
                status = " [INCREASED]"
            print(f"[НЕЯВ] Шаг {step_count_imp:4d}: t = {t_imp:.6f}, "
                  f"tau = {tau_imp:.6e}, err = {cur_err_imp:.2e}{status}")
        
        if t_imp >= T_final:
            finished_imp = True
            print(f"\n[НЕЯВ] Неявная схема завершена: {step_count_imp} шагов\n")
    
    # Обновление графиков
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
    
    # Остановка анимации когда обе схемы завершены
    if finished_exp and finished_imp:
        print("\n" + "="*70)
        print(f"ИТОГИ:")
        print(f"  Явная схема:   {step_count_exp} шагов")
        print(f"  Неявная схема: {step_count_imp} шагов")
        print(f"  Ускорение неявной схемы: {step_count_exp / step_count_imp:.2f}x")
        print("="*70)
        ani.event_source.stop()
    
    return (lines_dict['line_exp'], lines_dict['line_imp'], 
            texts_dict['text_exp'], texts_dict['text_imp'],
            lines_dict['line_tau_exp'], lines_dict['line_tau_imp'], 
            lines_dict['line_err_exp'], lines_dict['line_err_imp'])


# ==============================================================================
# ЗАПУСК АНИМАЦИИ
# ==============================================================================

# Создание анимации (максимум 2000 кадров, чтобы не зависнуть)
max_frames = 2000
ani = FuncAnimation(fig, update, frames=max_frames,
                    init_func=init, blit=False, interval=1, repeat=False)

plt.show()


# ==============================================================================
# СОХРАНЕНИЕ ИТОГОВЫХ РЕЗУЛЬТАТОВ
# ==============================================================================

save_final_results(
    x, y0, y_exp, y_imp,
    t_history_exp, t_history_imp,
    tau_history_exp, tau_history_imp,
    error_history_exp, error_history_imp,
    T_final, accuracy
)
