from explicit import explicit_step
from implicit import implicit_step
from utils import l2_norm

def adaptive_explicit_step(y, t, tau, h, max_sigma, accuracy, x, k_func, c_func, f_func, u1, u2):
    step_crushed = False
    step_increased = False

    # условие устойчивости: tau <= h^2 * min(c/k) / 2
    # используем max_sigma как ориентир, но адаптивность сама подстроит.
    
    while True:
        # Вычисляем решение с шагом tau (один шаг)
        y_tau = explicit_step(y, t, tau, h, x, k_func, c_func, f_func, u1, u2)
        
        # Вычисляем решение с шагом tau/2 (два шага)
        crush_tau = tau / 2.0
        
        y_crush = explicit_step(y, t, crush_tau, h, x, k_func, c_func, f_func, u1, u2)
        y_crush = explicit_step(y_crush, t + crush_tau, crush_tau, h, x, k_func, c_func, f_func, u1, u2)
        
        # Оценка ошибки по правилу Рунге
        cur_err = l2_norm(y_tau - y_crush)
        
        accept_decision = False
        
        if cur_err >= accuracy:
            if tau <= h**2 * 1e-4: # Защита от слишком малого шага
                accept_decision = True
            else:
                tau /= 2.0
                step_crushed = True
        else:
            if tau * 2.0 / h**2 > max_sigma:
                accept_decision = True
            else:
                tau *= 2.0
                step_increased = True
        
        if (step_crushed and step_increased) or accept_decision:
            return y_tau, tau, cur_err, step_crushed, step_increased


def adaptive_implicit_step(y, t, tau, h, accuracy, M, x, k_func, c_func, f_func, u1, u2):
    step_crushed = False
    step_increased = False
    
    while True:
        # Вычисляем решение с шагом tau (один шаг)
        y_tau = implicit_step(y, t, tau, h, x, k_func, c_func, f_func, u1, u2)
        
        # Вычисляем решение с шагом tau/2 (два шага)
        crush_tau = tau / 2.0
        
        y_crush = implicit_step(y, t, crush_tau, h, x, k_func, c_func, f_func, u1, u2)
        y_crush = implicit_step(y_crush, t + crush_tau, crush_tau, h, x, k_func, c_func, f_func, u1, u2)
        
        # Оценка ошибки по правилу Рунге
        cur_err = l2_norm(y_tau - y_crush)

        accept_decision = False
        
        if cur_err >= accuracy:
            if tau <= h**2 * 1e-5: 
                accept_decision = True
            else:
                tau /= 2.0
                step_crushed = True
        else:
            tau *= 2.0
            step_increased = True
        
        if (step_crushed and step_increased) or accept_decision:
            return y_tau, tau, cur_err, step_crushed, step_increased
