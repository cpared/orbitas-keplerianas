"""Módulo de lectura de datos."""
import numpy as np


def leer_archivo(arch):
    """Lee archivo de posiciones con formato: FECHA HORA POS_X POS_Y POS_Z VEL_X VEL_Y VEL_Z."""
    x, y, z, t = [], [], [], []
    with open(arch, "r") as f:
        step = int(arch.split("_")[-1].replace("s.txt", ""))
        for i, linea in enumerate(f):
            datos = linea.split()
            x.append(float(datos[2]))
            y.append(float(datos[3]))
            z.append(float(datos[4]))
            t.append(i * step)
    return np.array(t), np.array(x), np.array(y), np.array(z)
