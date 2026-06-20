"""Módulo de splines cúbicos naturales."""
import numpy as np


def thomas_tridiagonal(a, b, c, d):
    """Resuelve un sistema tridiagonal usando el algoritmo de Thomas."""
    n = len(d)
    b_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    b_prime[0] = b[0]
    d_prime[0] = d[0]
    
    for i in range(1, n):
        m = a[i] / b_prime[i - 1]
        b_prime[i] = b[i] - m * c[i - 1]
        d_prime[i] = d[i] - m * d_prime[i - 1]
    
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1] / b_prime[n - 1]
    
    for i in range(n - 2, -1, -1):
        x[i] = (d_prime[i] - c[i] * x[i + 1]) / b_prime[i]
    
    return x


def calcular_coeficientes(y, h, c):
    """Calcula los coeficientes (a, b, c, d) del spline cúbico."""
    n = len(y)
    a = y[:-1]
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c, d


def spline_natural(t, x):
    """Calcula los coeficientes de un spline cúbico natural."""
    n = len(t)
    h = t[1:] - t[:-1]

    a_diag = np.zeros(n)
    b_diag = np.zeros(n)
    c_diag = np.zeros(n)
    d_vec = np.zeros(n)

    b_diag[0] = 1.0
    b_diag[n-1] = 1.0

    for i in range(1, n-1):
        a_diag[i] = h[i-1]
        b_diag[i] = 2 * (h[i-1] + h[i])
        c_diag[i] = h[i]
        d_vec[i] = 3 * ((x[i+1] - x[i]) / h[i] - (x[i] - x[i-1]) / h[i-1])
    
    c = thomas_tridiagonal(a_diag, b_diag, c_diag, d_vec)
    return t, calcular_coeficientes(x, h, c)


def evaluar_spline_vectorizado(tiempos, t, a, b, c, d):
    """Evalúa el spline cúbico en múltiples instantes de tiempo."""
    scalar_input = np.isscalar(tiempos) or (isinstance(tiempos, np.ndarray) and tiempos.ndim == 0)
    if scalar_input:
        tiempos_arr = np.array([tiempos])
    else:
        tiempos_arr = np.array(tiempos)

    resultados = np.zeros_like(tiempos_arr, dtype=float)

    for j, tiempo in enumerate(tiempos_arr):
        i = np.searchsorted(t[:-1], tiempo, side='right') - 1
        i = max(0, min(i, len(t) - 2))
        dt = tiempo - t[i]
        resultados[j] = a[i] + b[i] * dt + c[i] * dt**2 + d[i] * dt**3

    if scalar_input:
        return resultados[0]
    return resultados
