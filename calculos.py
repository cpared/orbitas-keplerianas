"""Módulo de cálculos para los diferentes puntos del trabajo práctico."""
import numpy as np
from spline import spline_natural, evaluar_spline_vectorizado
from rk4 import propagar_orbita, calcular_energia, error_energia_relativo, calcular_momento_angular, calcular_derivada_momento_angular
from data import leer_archivo, leer_primer_estado


def calcular_orbita_spline(archivo_ref, archivo_muestras, tiempo_periodo=6600):
    """Calcula splines para las órbitas del archivo de muestras.
    
    Retorna tuplas de datos listos para graficar.
    """
    # Leer datos completos (1s) para splines de visualización
    t_ref, x_ref, y_ref, z_ref = leer_archivo(archivo_ref)
    
    t_fine = np.linspace(t_ref[0], t_ref[-1], min(len(t_ref), 5000))
    t_x, coef_x = spline_natural(t_ref, x_ref)
    t_y, coef_y = spline_natural(t_ref, y_ref)
    t_z, coef_z = spline_natural(t_ref, z_ref)
    
    x_spline = evaluar_spline_vectorizado(t_fine, t_x, *coef_x)
    y_spline = evaluar_spline_vectorizado(t_fine, t_y, *coef_y)
    z_spline = evaluar_spline_vectorizado(t_fine, t_z, *coef_z)
    
    # Leer datos de muestras (600s) y limitar al período
    tiempos, pos_x, pos_y, pos_z = leer_archivo(archivo_muestras)
    indices_periodo = tiempos <= tiempo_periodo
    tiempos_muestras = tiempos[indices_periodo]
    x_muestras = pos_x[indices_periodo]
    y_muestras = pos_y[indices_periodo]
    z_muestras = pos_z[indices_periodo]
    
    # Crear splines a partir de las muestras
    t_x_m, coef_x_m = spline_natural(tiempos_muestras, x_muestras)
    t_y_m, coef_y_m = spline_natural(tiempos_muestras, y_muestras)
    t_z_m, coef_z_m = spline_natural(tiempos_muestras, z_muestras)
    
    # Interpolar cada 60 segundos
    tiempos_interp = np.arange(tiempos_muestras[0], tiempos_muestras[-1] + 60, 60)
    x_interp = np.array([evaluar_spline_vectorizado(t, t_x_m, *coef_x_m) for t in tiempos_interp])
    y_interp = np.array([evaluar_spline_vectorizado(t, t_y_m, *coef_y_m) for t in tiempos_interp])
    z_interp = np.array([evaluar_spline_vectorizado(t, t_z_m, *coef_z_m) for t in tiempos_interp])
    
    return {
        'orbita_completa': (t_fine, x_spline, y_spline, z_spline),
        'orbita_periodo': (tiempos_interp, x_interp, y_interp, z_interp, tiempos_muestras, x_muestras, y_muestras, z_muestras),
        'componentes': (tiempos_interp, x_interp, y_interp, z_interp, tiempos_muestras, x_muestras, y_muestras, z_muestras),
        'splines': (t_x_m, coef_x_m, t_y_m, coef_y_m, t_z_m, coef_z_m)
    }


def calcular_error_spline(spl_x, spl_y, spl_z, archivo_ref, t_max):
    """Calcula errores entre spline (600s) y referencia (1s)."""
    t_x, a_x, b_x, c_x, d_x = spl_x
    t_y, a_y, b_y, c_y, d_y = spl_y
    t_z, a_z, b_z, c_z, d_z = spl_z

    t_ref, x_ref, y_ref, z_ref = leer_archivo(archivo_ref)
    
    indices = t_ref <= t_max
    t_ref = t_ref[indices]
    x_ref = x_ref[indices]
    y_ref = y_ref[indices]
    z_ref = z_ref[indices]

    x_spline = evaluar_spline_vectorizado(t_ref, t_x, a_x, b_x, c_x, d_x)
    y_spline = evaluar_spline_vectorizado(t_ref, t_y, a_y, b_y, c_y, d_y)
    z_spline = evaluar_spline_vectorizado(t_ref, t_z, a_z, b_z, c_z, d_z)

    dx = x_spline - x_ref
    dy = y_spline - y_ref
    dz = z_spline - z_ref
    
    error_norma = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return {
        'tiempos': t_ref,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'error_max_x': np.max(np.abs(dx)),
        'error_max_y': np.max(np.abs(dy)),
        'error_max_z': np.max(np.abs(dz)),
        'error_max_posicion': np.max(error_norma)
    }


def calcular_orbitas_rk4(condiciones_iniciales, h=1):
    """Calcula órbitas propagadas con RK4.
    
    condiciones_iniciales: lista de (nombre, r0, v0, T)
    Retorna posiciones, velocidades y energías.
    """
    resultados = []
    for nombre, r0, v0, T in condiciones_iniciales:
        pos, vel = propagar_orbita(r0, v0, h, T)
        energia = calcular_energia(pos, vel)
        error_rel = error_energia_relativo(energia)
        
        resultados.append({
            'nombre': nombre,
            'posiciones': pos,
            'velocidades': vel,
            'energia': energia,
            'error_energia': error_rel
        })
    
    return resultados


def comparar_rk4_con_spline(archivo_600s, datos_spline, tiempo_periodo=6600):
    """Punto D: compara RK4(600s)+spline y RK4(1s) vs spline de datos reales."""
    r0, v0 = leer_primer_estado(archivo_600s)

    pos_600s, _ = propagar_orbita(r0, v0, 600, tiempo_periodo)
    tiempos_600s = np.arange(0, tiempo_periodo + 600, 600)
    t_x_m, coef_x_m = spline_natural(tiempos_600s, pos_600s[:, 0])
    t_y_m, coef_y_m = spline_natural(tiempos_600s, pos_600s[:, 1])
    t_z_m, coef_z_m = spline_natural(tiempos_600s, pos_600s[:, 2])

    tiempos_finos = np.arange(0, tiempo_periodo + 60, 60)
    x_rk4_600s = np.array([evaluar_spline_vectorizado(t, t_x_m, *coef_x_m) for t in tiempos_finos])
    y_rk4_600s = np.array([evaluar_spline_vectorizado(t, t_y_m, *coef_y_m) for t in tiempos_finos])
    z_rk4_600s = np.array([evaluar_spline_vectorizado(t, t_z_m, *coef_z_m) for t in tiempos_finos])

    pos_1s, _ = propagar_orbita(r0, v0, 1, tiempo_periodo)
    tiempos_1s = np.arange(0, tiempo_periodo + 1, 1)
    t_x_1s, coef_x_1s = spline_natural(tiempos_1s, pos_1s[:, 0])
    t_y_1s, coef_y_1s = spline_natural(tiempos_1s, pos_1s[:, 1])
    t_z_1s, coef_z_1s = spline_natural(tiempos_1s, pos_1s[:, 2])

    x_rk4_1s = np.array([evaluar_spline_vectorizado(t, t_x_1s, *coef_x_1s) for t in tiempos_finos])
    y_rk4_1s = np.array([evaluar_spline_vectorizado(t, t_y_1s, *coef_y_1s) for t in tiempos_finos])
    z_rk4_1s = np.array([evaluar_spline_vectorizado(t, t_z_1s, *coef_z_1s) for t in tiempos_finos])

    t_ref, x_ref, y_ref, z_ref, _, _, _, _ = datos_spline['orbita_periodo']

    norma_rk4_600s = np.sqrt(x_rk4_600s**2 + y_rk4_600s**2 + z_rk4_600s**2)
    norma_rk4_1s = np.sqrt(x_rk4_1s**2 + y_rk4_1s**2 + z_rk4_1s**2)
    norma_ref = np.sqrt(x_ref**2 + y_ref**2 + z_ref**2)

    return {
        'tiempos': tiempos_finos,
        'norma_rk4_600s': norma_rk4_600s,
        'norma_rk4_1s': norma_rk4_1s,
        'norma_ref': norma_ref,
        'diff_norma': norma_rk4_600s - norma_ref
    }


def calcular_magnitudes_24h(condiciones_iniciales, h=1, tiempo_total=86400):
    """Propaga 24h y calcula |h|, |ḣ|, ε para cada órbita."""
    resultados = []
    for nombre, r0, v0, T in condiciones_iniciales:
        pos, vel = propagar_orbita(r0, v0, h, tiempo_total)
        h_vec = calcular_momento_angular(pos, vel)
        hdot_vec = calcular_derivada_momento_angular(pos, vel)
        energia = calcular_energia(pos, vel)
        tiempos = np.arange(0, tiempo_total + h, h)
        resultados.append({
            'nombre': nombre,
            'tiempos': tiempos,
            'h_mag': np.linalg.norm(h_vec, axis=1),
            'hdot_mag': np.linalg.norm(hdot_vec, axis=1),
            'energia': energia
        })
    return resultados
