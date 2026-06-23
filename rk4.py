"""Módulo de propagación de órbitas con RK4."""
import numpy as np

MU = 398600.4418  # Parámetro gravitacional terrestre [km^3/s^2]


def fr(r, v):
    """Derivada de posición = velocidad."""
    return v


def fv(r):
    """Derivada de velocidad = aceleración gravitatoria."""
    r_norm = np.linalg.norm(r)
    return -(MU / r_norm**3) * r


def paso_rk4(ri, vi, h):
    """Calcula el estado (ri+1, vi+1) a partir del estado (ri, vi) usando RK4."""
    k1r = fr(ri, vi)
    k1v = fv(ri)

    k2r = fr(ri + 0.5 * h * k1r, vi + 0.5 * h * k1v)
    k2v = fv(ri + 0.5 * h * k1r)

    k3r = fr(ri + 0.5 * h * k2r, vi + 0.5 * h * k2v)
    k3v = fv(ri + 0.5 * h * k2r)

    k4r = fr(ri + h * k3r, vi + h * k3v)
    k4v = fv(ri + h * k3r)

    ri_1 = ri + (h / 6) * (k1r + 2*k2r + 2*k3r + k4r)
    vi_1 = vi + (h / 6) * (k1v + 2*k2v + 2*k3v + k4v)

    return ri_1, vi_1


def propagar_orbita(r0, v0, h, tiempo_total):
    """Propaga la órbita desde el estado inicial durante tiempo_total segundos."""
    n_pasos = int(tiempo_total / h)

    posiciones = np.zeros((n_pasos + 1, 3))
    velocidades = np.zeros((n_pasos + 1, 3))

    posiciones[0] = r0
    velocidades[0] = v0

    for i in range(n_pasos):
        posiciones[i+1], velocidades[i+1] = paso_rk4(posiciones[i], velocidades[i], h)

    return posiciones, velocidades


def calcular_energia(posiciones, velocidades):
    """Calcula la energía orbital para cada paso."""
    energia = 0.5 * np.sum(velocidades**2, axis=1) - MU / np.linalg.norm(posiciones, axis=1)
    return energia


def error_energia_relativo(energia):
    """Calcula el error relativo de energía respecto al valor inicial."""
    return np.abs((energia - energia[0]) / energia[0])


def calcular_momento_angular(posiciones, velocidades):
    """Calcula el momento angular específico h = r × v para cada paso."""
    return np.cross(posiciones, velocidades)


def calcular_derivada_momento_angular(posiciones, velocidades):
    """Calcula la derivada del momento angular ḣ = r × r̈ para cada paso."""
    r_norm = np.linalg.norm(posiciones, axis=1)
    aceleracion = -(MU / r_norm**3).reshape(-1, 1) * posiciones
    return np.cross(posiciones, aceleracion)
