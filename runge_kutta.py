import numpy as np
from scipy.integrate import solve_ivp

# выражение для dY/dt = f(t, Y)
def equation_rhs(t, y, h):
    dydt = np.zeros_like(y)

    y_left_boundary = 0
    y_right_boundary = 0
    
    dydt[0] = (y_left_boundary - 2*y[0] + y[1]) / h**2

    for j in range(1, len(y) - 1):
        dydt[j] = (y[j-1] - 2*y[j] + y[j+1]) / h**2

    dydt[-1] = (y[-2] - 2*y[-1] + y_right_boundary) / h**2
    
    return dydt

def solve_runge_kutta(y0, t, N, h):
    sol = solve_ivp(
        fun=equation_rhs, 
        t_span=[0, t], 
        y0=y0[1:N], 
        args=(h,), 
        dense_output=True,
        rtol=1e-6, 
        atol=1e-9,
    )
    
    y_internal = sol.sol(t)
    y = np.zeros(N + 1)
    y[1:N] = y_internal
    
    return y
