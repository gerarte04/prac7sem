import numpy as np

def tridiagonal_solve(a, b, c, d):
    n = len(b)
    cp = np.zeros(n-1)
    dp = np.zeros(n)
    x = np.zeros(n)
    
    # прямой ход
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])
    
    # обратный ход
    x[n-1] = dp[n-1]
    for i in reversed(range(n-1)):
        x[i] = dp[i] - cp[i] * x[i+1]
    
    return x
