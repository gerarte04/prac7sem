import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_animation_figure(x, y0, tau, T_final, accuracy):
    fig = plt.figure(figsize=(16, 9))
    
    # График решения
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('Решение: Численное vs Точное')
    # ax1.set_ylim(-0.05, 1.05) # Limits might change for nonlinear task
    ax1.grid(True, alpha=0.3)
    line_num, = ax1.plot(x, y0, 'r-', linewidth=2, label='Численное')
    line_exact, = ax1.plot(x, y0, 'b--', linewidth=1.5, label='Точное')
    ax1.legend(loc='upper right')
    text_info = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontfamily='monospace',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # График профиля ошибки
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|u_num - u_exact|')
    ax2.set_title('Профиль ошибки')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    line_err_prof, = ax2.plot(x, np.zeros_like(x) + 1e-16, 'g-', linewidth=2)
    
    # График истории шага tau
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('dt')
    ax3.set_title('Адаптивный шаг времени')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    line_dt, = ax3.plot([], [], 'r.-', markersize=3, label='dt')
    ax3.set_xlim(0, T_final)
    # ax3.set_ylim(tau * 0.5, tau * 100) # Let it autoscale or set dynamic
    
    # График истории ошибки (Рунге)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_xlabel('t')
    ax4.set_ylabel('Ошибка (Runge est.)')
    ax4.set_title('Контроль ошибки')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    line_err_hist, = ax4.plot([], [], 'r.-', markersize=3, label='Error est.')
    ax4.axhline(y=accuracy, color='k', linestyle='--', linewidth=1, 
                label=f'Требуемая точность', alpha=0.7)
    ax4.legend()
    ax4.set_xlim(0, T_final)
    ax4.set_ylim(accuracy * 0.01, accuracy * 100)
    
    plt.tight_layout()
    
    axes_dict = {
        'ax1': ax1, 'ax2': ax2, 'ax3': ax3, 'ax4': ax4
    }
    
    lines_dict = {
        'line_num': line_num,
        'line_exact': line_exact,
        'line_err_prof': line_err_prof,
        'line_dt': line_dt,
        'line_err_hist': line_err_hist
    }
    
    texts_dict = {
        'text_info': text_info
    }
    
    return fig, axes_dict, lines_dict, texts_dict


def update_animation_plots(lines_dict, texts_dict, 
                           u, u_exact,
                           t, dt,
                           cur_err,
                           step_count,
                           t_history,
                           dt_history,
                           error_history,
                           finished,
                           T_final):
    # Обновление графиков решений
    lines_dict['line_num'].set_ydata(u)
    lines_dict['line_exact'].set_ydata(u_exact)
    
    # Автомасштабирование y для решения (так как оно растет)
    ax1 = lines_dict['line_num'].axes
    max_val = max(np.max(u), np.max(u_exact))
    if max_val > ax1.get_ylim()[1] or max_val < ax1.get_ylim()[1] * 0.5:
        ax1.set_ylim(-0.1 * max_val, 1.1 * max_val)

    # Обновление профиля ошибки
    err_prof = np.abs(u - u_exact)
    lines_dict['line_err_prof'].set_ydata(err_prof)
    ax2 = lines_dict['line_err_prof'].axes
    max_err = np.max(err_prof)
    if max_err > 0:
        ax2.set_ylim(max(1e-16, max_err * 1e-4), max_err * 2)

    # Обновление текстовой информации
    status_str = "ЗАВЕРШЕНО" if finished else "Выполняется"
    texts_dict['text_info'].set_text(f'Шаг: {step_count}\n'
                                    f't = {t:.5f}\n'
                                    f'dt = {dt:.2e}\n'
                                    f'err = {cur_err:.2e}\n'
                                    f'{status_str}')
    
    # Обновление графиков истории
    if len(t_history) > 1:
        lines_dict['line_dt'].set_data(t_history[:-1], dt_history[:-1])
        lines_dict['line_err_hist'].set_data(t_history[1:], error_history[1:])
        
        ax3 = lines_dict['line_dt'].axes
        ax3.set_xlim(0, max(T_final, t))
        min_dt = np.min(dt_history)
        max_dt = np.max(dt_history)
        ax3.set_ylim(min_dt * 0.5, max_dt * 2)

        ax4 = lines_dict['line_err_hist'].axes
        ax4.set_xlim(0, max(T_final, t))


def save_final_results(x, u, u_exact,
                       t_history,
                       dt_history,
                       error_history,
                       T_final, accuracy, filename='nonlinear_results.png'):
    print("\nСохранение итоговых графиков...")
    fig_final = plt.figure(figsize=(16, 10))
    
    # График решений
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, u, 'r-', label='Численное', linewidth=2)
    ax1.plot(x, u_exact, 'b--', label='Точное', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title(f'Решение в момент t = {T_final:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График разности решений
    ax2 = plt.subplot(2, 2, 2)
    diff = np.abs(u - u_exact)
    ax2.plot(x, diff, 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|u - u_exact|')
    ax2.set_title(f'Разность решений (max = {np.max(diff):.2e})')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # График изменения шага dt
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t_history[:-1], dt_history[:-1], 'r.-', label='dt', markersize=3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('dt')
    ax3.set_title('Адаптивный шаг времени')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # График ошибки
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t_history[1:], error_history[1:], 'r.-', markersize=3, label='Error est.')
    ax4.axhline(y=accuracy, color='k', linestyle='--', label=f'accuracy={accuracy}')
    ax4.set_xlabel('t')
    ax4.set_ylabel('Ошибка (Runge)')
    ax4.set_title('Контроль ошибки')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Графики сохранены в '{filename}'")
    plt.close(fig_final)
