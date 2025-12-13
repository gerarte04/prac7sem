from explicit import explicit_step
from implicit import implicit_step
from utils import build_implicit_matrix, l2_norm

def adaptive_explicit_step(y, tau, h, max_sigma, accuracy):
    step_crushed = False
    step_increased = False
    
    while True:
        sigma = tau / h**2
        # Вычисляем решение с шагом tau (один шаг)
        y_tau = explicit_step(y, sigma)
        
        # Вычисляем решение с шагом tau/2 (два шага)
        crush_tau = tau / 2.0
        crush_sigma = crush_tau / h**2
        y_crush = explicit_step(y, crush_sigma)
        y_crush = explicit_step(y_crush, crush_sigma)
        
        # Оценка ошибки по правилу Рунге
        cur_err = l2_norm(y_tau - y_crush)
        
        accept_decision = False
        
        if cur_err >= accuracy:
            # Ошибка слишком большая
            if tau <= h**2:
                # Шаг уже минимальный - принимаем решение
                accept_decision = True
            else:
                # Уменьшаем шаг и продолжаем
                tau /= 2.0
                step_crushed = True
        else:
            # Ошибка приемлемая
            if tau * 2.0 / h**2 > max_sigma:
                # Не можем больше увеличивать шаг - принимаем решение
                accept_decision = True
            else:
                # Увеличиваем шаг и продолжаем
                tau *= 2.0
                step_increased = True
        
        # Принимаем решение
        if (step_crushed and step_increased) or accept_decision:
            return y_tau, tau, cur_err, step_crushed, step_increased


def adaptive_implicit_step(y, tau, h, accuracy, M):
    step_crushed = False
    step_increased = False
    
    while True:
        sigma = tau / h**2
        # Вычисляем решение с шагом tau (один шаг)
        A_tau = build_implicit_matrix(sigma, M)
        y_tau = implicit_step(y, A_tau)
        
        # Вычисляем решение с шагом tau/2 (два шага)
        crush_tau = tau / 2.0
        crush_sigma = crush_tau / h**2
        A_crush = build_implicit_matrix(crush_sigma, M)
        y_crush = implicit_step(y, A_crush)
        y_crush = implicit_step(y_crush, A_crush)
        
        # Оценка ошибки по правилу Рунге
        cur_err = l2_norm(y_tau - y_crush)

        accept_decision = False
        
        if cur_err >= accuracy:
            # Ошибка слишком большая
            if tau <= h**2:
                # Шаг уже минимальный - принимаем решение
                accept_decision = True
            else:
                # Уменьшаем шаг и продолжаем
                tau /= 2.0
                step_crushed = True
        else:
            # Ошибка приемлемая - увеличиваем шаг и продолжаем
            tau *= 2.0
            step_increased = True
        
        # Принимаем решение
        if (step_crushed and step_increased) or accept_decision:
            return y_tau, tau, cur_err, step_crushed, step_increased
