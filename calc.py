import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from explicit import explicit_step
from implicit import implicit_step

L = 1.0       # Длина области
Nx = 50       # Количество *интервалов* по x (точек будет Nx + 1)
h = L / Nx    # Шаг по x
x = np.linspace(0, L, Nx + 1) # Массив узлов по x

# Для правила Рунге для явной схемы нужно, чтобы 2*tau было устойчиво.
# sigma_2tau = 2*tau/h**2 <= 0.5  =>  tau/h**2 <= 0.25
sigma = 0.25
tau = sigma * h**2  # Шаг по времени
T_final = 0.1       # Конечное время
N_steps = int(T_final / tau) # Количество шагов

print(f"Параметры сетки:")
print(f"Nx = {Nx}, h = {h:.4f}")
print(f"N_steps = {N_steps}, tau = {tau:.6f}")
print(f"sigma = {sigma:.3f} (устойчиво для явной схемы)")

y0 = np.exp(-100 * (x - L/2)**2)
y0[0] = 0
y0[-1] = 0

y_exp = y0.copy()
y_imp = y0.copy()
y_exp_2tau = y0.copy()
y_imp_2tau = y0.copy()


M = Nx - 1 # Размерность СЛАУ (количество внутренних точек)

# Матрица A для шага tau (sigma)
sigma_imp_1tau = sigma
A_imp_1tau = np.diag((1 + 2 * sigma_imp_1tau) * np.ones(M)) + \
             np.diag(-sigma_imp_1tau * np.ones(M - 1), k=1) + \
             np.diag(-sigma_imp_1tau * np.ones(M - 1), k=-1)

# Матрица A для шага 2*tau (2*sigma)
sigma_imp_2tau = 2 * sigma
A_imp_2tau = np.diag((1 + 2 * sigma_imp_2tau) * np.ones(M)) + \
             np.diag(-sigma_imp_2tau * np.ones(M - 1), k=1) + \
             np.diag(-sigma_imp_2tau * np.ones(M - 1), k=-1)

# Параметры sigma для явной схемы
sigma_exp_1tau = sigma
sigma_exp_2tau = 2 * sigma

# ------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Решение y_t = y_xx (sigma = {sigma})", fontsize=16)

ax1.set_title("Явная схема")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, linestyle='--', alpha=0.6)
line1, = ax1.plot(x, y0, 'r-', label='Явная (τ)')
line1_2tau, = ax1.plot(x, y0, 'r:', alpha=0.5, label='Явная (2τ)')
ax1.legend(loc='upper right')
time_text = ax1.text(0.02, 0.83, '', transform=ax1.transAxes)
runge_text_exp = ax1.text(0.02, 0.93, '', transform=ax1.transAxes, 
                          fontfamily='monospace')

ax2.set_title("Неявная схема")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True, linestyle='--', alpha=0.6)
line2, = ax2.plot(x, y0, 'b-', label='Неявная (τ)')
line2_2tau, = ax2.plot(x, y0, 'b:', alpha=0.5, label='Неявная (2τ)')
ax2.legend(loc='upper right')
runge_text_imp = ax2.text(0.02, 0.93, '', transform=ax2.transAxes, 
                          fontfamily='monospace')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

runge_log = {
    "exp": "Оценка по Рунге (Явная): --",
    "imp": "Оценка по Рунге (Неявная): --"
}

def init():
    line1.set_ydata(y0)
    line2.set_ydata(y0)
    line1_2tau.set_ydata(y0)
    line2_2tau.set_ydata(y0)
    time_text.set_text('')
    runge_text_exp.set_text(runge_log["exp"])
    runge_text_imp.set_text(runge_log["imp"])
    return line1, line2, line1_2tau, line2_2tau, time_text, \
           runge_text_exp, runge_text_imp

def update(n):
    global y_exp, y_imp, y_exp_2tau, y_imp_2tau
    
    # Шаг на сетке tau
    y_exp = explicit_step(y_exp, sigma_exp_1tau)
    y_imp = implicit_step(y_imp, A_imp_1tau)
    
    # Каждые 2 шага делаем шаг на сетке 2*tau (на нечетных n)
    if n % 2 == 1:
        y_exp_2tau = explicit_step(y_exp_2tau, sigma_exp_2tau)
        y_imp_2tau = implicit_step(y_imp_2tau, A_imp_2tau)

    line1.set_ydata(y_exp)
    line2.set_ydata(y_imp)
    if n % 2 == 1:
        line1_2tau.set_ydata(y_exp_2tau)
        line2_2tau.set_ydata(y_imp_2tau)

    time_text.set_text(f"Шаг: {n+1} / {N_steps}\nВремя: {(n+1) * tau:.4f} c")
    
    # Оценка по Рунге (каждые 50 итераций)
    # (сравниваем на нечетном шаге n, когда обе сетки совпали по времени)
    if (n + 1) % 50 == 0 and n % 2 == 1:
        # Ошибка = ||y_tau - y_2tau||_inf (максимальная разница)
        # p=1, поэтому Error = y_tau - y_2tau
        error_exp = np.linalg.norm(y_exp - y_exp_2tau, np.inf)
        error_imp = np.linalg.norm(y_imp - y_imp_2tau, np.inf)
        
        runge_log["exp"] = f"Оценка (Явная): {error_exp:.2e}"
        runge_log["imp"] = f"Оценка (Неявная): {error_imp:.2e}"
        
        runge_text_exp.set_text(runge_log["exp"])
        runge_text_imp.set_text(runge_log["imp"])

    elif (n+1) % 50 != 0:
        runge_text_exp.set_text(runge_log["exp"])
        runge_text_imp.set_text(runge_log["imp"])

    return line1, line2, line1_2tau, line2_2tau, time_text, \
           runge_text_exp, runge_text_imp

ani = FuncAnimation(fig, update, frames=N_steps,
                    init_func=init, blit=True, interval=20, repeat=False)

plt.show()
