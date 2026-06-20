"""Módulo de graficación de órbitas y datos."""
import numpy as np
import matplotlib.pyplot as plt


def plot_orbita_3d(x, y, z, titulo="Órbita 3D"):
    """Grafica una trayectoria tridimensional de una órbita."""
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title(titulo)
    plt.show()


def plot_orbita_3d_con_spline(x_interp, y_interp, z_interp, x_muestras, y_muestras, z_muestras, titulo="", save_path=None):
    """Grafica órbita interpolada con spline y muestras originales."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_interp, y_interp, z_interp, label="Spline interpolada")
    ax.scatter(x_muestras, y_muestras, z_muestras, color="red", s=60, label="Muestras")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_componentes_vs_tiempo(tiempos, x_interp, y_interp, z_interp, tiempos_muestras, x_muestras, y_muestras, z_muestras):
    """Grafica componentes X, Y, Z vs tiempo."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(tiempos, x_interp)
    axs[0].scatter(tiempos_muestras, x_muestras, color="red")
    axs[0].set_ylabel("X [km]")
    axs[0].grid()

    axs[1].plot(tiempos, y_interp)
    axs[1].scatter(tiempos_muestras, y_muestras, color="red")
    axs[1].set_ylabel("Y [km]")
    axs[1].grid()

    axs[2].plot(tiempos, z_interp)
    axs[2].scatter(tiempos_muestras, z_muestras, color="red")
    axs[2].set_ylabel("Z [km]")
    axs[2].set_xlabel("Tiempo [s]")
    axs[2].grid()

    plt.tight_layout()
    plt.show()


def plot_error_spline(tiempos, dx, dy, dz):
    """Grafica los errores en cada componente."""
    plt.figure(figsize=(10, 6))
    plt.plot(tiempos, dx, label="ΔX")
    plt.plot(tiempos, dy, label="ΔY")
    plt.plot(tiempos, dz, label="ΔZ")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error [km]")
    plt.title("Diferencias entre spline y referencia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_conservacion_energia(errores_relativos, nombres_orbitas):
    """Grafica el error relativo de energía para múltiples órbitas."""
    plt.figure(figsize=(10, 6))
    for nombre, error in zip(nombres_orbitas, errores_relativos):
        plt.plot(error, label=nombre)
    plt.xlabel("Paso temporal")
    plt.ylabel("Error relativo de energía")
    plt.title("Conservación de energía con RK4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_orbitas_3d_multiples(posiciones_list, nombres_orbitas, posiciones_iniciales, titulo="", save_path=None):
    """Grafica múltiples órbitas en 3D con marcadores de inicio."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colores = ['blue', 'orange', 'green']
    for pos, nombre, color in zip(posiciones_list, nombres_orbitas, colores):
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=nombre)

    for pos_init, color in zip(posiciones_iniciales, colores):
        ax.scatter(*pos_init, s=50, zorder=5, color=color)

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    # Dibujar superficie terrestre
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R = 6378
    x = R*np.outer(np.cos(u), np.sin(v))
    y = R*np.outer(np.sin(u), np.sin(v))
    z = R*np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.3)
    
    plt.show()
