import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from explicit import explicit_step
from implicit import implicit_step

L = 2.0         # длина стержня
T = 0.2         # общее время симуляции

N = 100         # количество пространственных узлов
M = 2000        # количество временных шагов

h = L / N       # шаг по пространству
tau = T / M     # щаг по времени

k = 1.0         # коэффициент диффузии

# условие устойчивости
sigma = k * tau / h**2
if sigma > 0.5:
    print(f"WARNING! sigma = {sigma:.2f} > 0.5, explicit method'll be unstable")
    # os._exit(1)

print(f"sigma = {sigma:.4f}")

x = np.linspace(0, L, N + 1)
y0 = np.exp(-100 * (x - L/2)**2)

y_ex = y0.copy()
y_im = y0.copy()

fig, axs = plt.subplots(1, 2)
axs[0].set_title("Явный метод для $y_t = y_{xx}$")
axs[1].set_title("Неявный метод для $y_t = y_{xx}$")

for i in range(2):
    axs[i].set_xlabel("Координата x")
    axs[i].set_ylabel("Значение y")
    axs[i].set_xlim(0, L)
    axs[i].set_ylim(0, 1.1)
    axs[i].grid(True)

line_ex, = axs[0].plot(x, y_ex, color='b', lw=2)
line_im, = axs[1].plot(x, y_im, color='g', lw=2)
time_text = axs[0].text(0.05, 0.9, '', transform=axs[0].transAxes)

def animate(j):
    global y_ex, y_im

    y_ex = explicit_step(y_ex, sigma, N)
    y_im = implicit_step(y_im, sigma, N)
    
    line_ex.set_ydata(y_ex)
    line_im.set_ydata(y_im)
    time_text.set_text(f'Время t = {j * tau:.3f}')
    
    return line_ex, line_im, time_text

ani = FuncAnimation(fig, animate, frames=M, interval=10, blit=True, repeat=False)
plt.show()
