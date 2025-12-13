import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_animation_figure(x, y0, tau, T_final, accuracy):
    fig = plt.figure(figsize=(16, 9))
    
    # График явной схемы
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Явная схема')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    line_exp, = ax1.plot(x, y0, 'r-', linewidth=2, label='Явная')
    ax1.plot(x, y0, 'k--', alpha=0.3, linewidth=1, label='Нач. усл.')
    ax1.legend(loc='upper right')
    text_exp = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontfamily='monospace',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # График неявной схемы
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Неявная схема')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    line_imp, = ax2.plot(x, y0, 'b-', linewidth=2, label='Неявная')
    ax2.plot(x, y0, 'k--', alpha=0.3, linewidth=1, label='Нач. усл.')
    ax2.legend(loc='upper right')
    text_imp = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                        verticalalignment='top', fontfamily='monospace',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # График истории шага tau
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('tau')
    ax3.set_title('Адаптивный шаг времени')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    line_tau_exp, = ax3.plot([], [], 'r.-', markersize=3, label='Явная')
    line_tau_imp, = ax3.plot([], [], 'b.-', markersize=3, label='Неявная')
    ax3.legend()
    ax3.set_xlim(0, T_final)
    ax3.set_ylim(tau * 0.5, tau * 100)
    
    # График истории ошибок
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_xlabel('t')
    ax4.set_ylabel('Ошибка (L2 норма)')
    ax4.set_title('Контроль ошибки по правилу Рунге')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    line_err_exp, = ax4.plot([], [], 'r.-', markersize=3, label='Явная')
    line_err_imp, = ax4.plot([], [], 'b.-', markersize=3, label='Неявная')
    ax4.axhline(y=accuracy, color='k', linestyle='--', linewidth=1, 
                label=f'Требуемая точность', alpha=0.7)
    ax4.legend()
    ax4.set_xlim(0, T_final)
    ax4.set_ylim(accuracy * 0.1, accuracy * 10)
    
    plt.tight_layout()
    
    axes_dict = {
        'ax1': ax1, 'ax2': ax2, 'ax3': ax3, 'ax4': ax4
    }
    
    lines_dict = {
        'line_exp': line_exp,
        'line_imp': line_imp,
        'line_tau_exp': line_tau_exp,
        'line_tau_imp': line_tau_imp,
        'line_err_exp': line_err_exp,
        'line_err_imp': line_err_imp
    }
    
    texts_dict = {
        'text_exp': text_exp,
        'text_imp': text_imp
    }
    
    return fig, axes_dict, lines_dict, texts_dict


def update_animation_plots(lines_dict, texts_dict, 
                           y_exp, y_imp, 
                           t_exp, t_imp,
                           tau_exp, tau_imp,
                           cur_err_exp, cur_err_imp,
                           step_count_exp, step_count_imp,
                           t_history_exp, t_history_imp,
                           tau_history_exp, tau_history_imp,
                           error_history_exp, error_history_imp,
                           finished_exp, finished_imp,
                           T_final):
    # Обновление графиков решений
    lines_dict['line_exp'].set_ydata(y_exp)
    lines_dict['line_imp'].set_ydata(y_imp)
    
    # Обновление текстовой информации
    status_str_exp = "ЗАВЕРШЕНО" if finished_exp else "Выполняется"
    texts_dict['text_exp'].set_text(f'Шаг: {step_count_exp}\n'
                                    f't = {t_exp:.5f}\n'
                                    f'τ = {tau_exp:.2e}\n'
                                    f'err = {cur_err_exp:.2e}\n'
                                    f'{status_str_exp}')
    
    status_str_imp = "ЗАВЕРШЕНО" if finished_imp else "Выполняется"
    texts_dict['text_imp'].set_text(f'Шаг: {step_count_imp}\n'
                                    f't = {t_imp:.5f}\n'
                                    f'τ = {tau_imp:.2e}\n'
                                    f'err = {cur_err_imp:.2e}\n'
                                    f'{status_str_imp}')
    
    # Обновление графиков истории
    if len(t_history_exp) > 1:
        lines_dict['line_tau_exp'].set_data(t_history_exp[:-1], tau_history_exp[:-1])
        lines_dict['line_err_exp'].set_data(t_history_exp[1:], error_history_exp[1:])
    
    if len(t_history_imp) > 1:
        lines_dict['line_tau_imp'].set_data(t_history_imp[:-1], tau_history_imp[:-1])
        lines_dict['line_err_imp'].set_data(t_history_imp[1:], error_history_imp[1:])


def save_final_results(x, y0, y_exp, y_imp, 
                       t_history_exp, t_history_imp,
                       tau_history_exp, tau_history_imp,
                       error_history_exp, error_history_imp,
                       T_final, accuracy, filename='adaptive_step_results.png'):
    print("\nСохранение итоговых графиков...")
    fig_final = plt.figure(figsize=(16, 10))
    
    # График решений
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x, y0, 'k--', label='Начальное условие', alpha=0.5)
    ax1.plot(x, y_exp, 'r-', label='Явная схема', linewidth=2)
    ax1.plot(x, y_imp, 'b-', label='Неявная схема', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Решение в момент t = {T_final}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График разности решений
    ax2 = plt.subplot(2, 3, 2)
    diff = np.abs(y_exp - y_imp)
    ax2.plot(x, diff, 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|y_exp - y_imp|')
    ax2.set_title(f'Разность решений (max = {np.max(diff):.2e})')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # График изменения шага tau для явной схемы
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t_history_exp[:-1], tau_history_exp[:-1], 'r.-', label='Явная', markersize=3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('tau')
    ax3.set_title('Адаптивный шаг (Явная схема)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # График изменения шага tau для неявной схемы
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t_history_imp[:-1], tau_history_imp[:-1], 'b.-', label='Неявная', markersize=3)
    ax4.set_xlabel('t')
    ax4.set_ylabel('tau')
    ax4.set_title('Адаптивный шаг (Неявная схема)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # График ошибки для явной схемы
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t_history_exp[1:], error_history_exp[1:], 'r.-', markersize=3)
    ax5.axhline(y=accuracy, color='k', linestyle='--', label=f'accuracy={accuracy}')
    ax5.set_xlabel('t')
    ax5.set_ylabel('Ошибка (L2 норма)')
    ax5.set_title('Контроль ошибки (Явная)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # График ошибки для неявной схемы
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(t_history_imp[1:], error_history_imp[1:], 'b.-', markersize=3)
    ax6.axhline(y=accuracy, color='k', linestyle='--', label=f'accuracy={accuracy}')
    ax6.set_xlabel('t')
    ax6.set_ylabel('Ошибка (L2 норма)')
    ax6.set_title('Контроль ошибки (Неявная)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Графики сохранены в '{filename}'")
    plt.close(fig_final)
