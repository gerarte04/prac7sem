import numpy as np

def explicit_step(y, sigma, N):
    y_new = y.copy()
    y_new[1:-1] = y[1:-1] + sigma * (y[2:] - 2*y[1:-1] + y[:-2]) # явная схема для i = 1, .., N - 1
    
    return y_new
