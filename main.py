import numpy as np
import matplotlib.pyplot as plt

ARCHIVO_1_SEG = "SACD_TPV_step_1s.txt"
ARCHIVO_60_SEG = "SACD_TPV_step_60s.txt"
ARCHIVO_600_SEG = "SACD_TPV_step_600s.txt"

# Parámetro gravitacional terrestre [km^3/s^2]
MU = 398600.4418

# CONDICIONES INICIALES
# Órbita circular
P0_circular = np.array([6125.24, -3547.04, -3.31277])
V0_circular = np.array([-0.519174, -0.903482, 7.43159])

# Órbita geoestacionaria
P0_geo = np.array([-41974.9, 4012.93, 23.5515])
V0_geo = np.array([-0.292606, -3.06063, 0.000293508])

# Órbita Molniya
P0_molniya = np.array([305.926, 3162.83, 6324.79])
V0_molniya = np.array([-9.86976, 0.326984, 0.313879])

def leer_archivo(arch):
  """
  Lee el archivo con el formato:

  FECHA HORA POS_X POS_Y POS_Z VEL_X VEL_Y VEL_Z

  y guarda los valores de las posiciones X,Y,Z en las variables x,y,z.
  Adicionalmente guarda tambien la cantidad de registros en la variable t.
  Si el archivo no existe arroja un error.
  """
  x = []
  y = []
  z = []
  t = []
  with open(arch, "r") as archivo_abierto:
      step = int(arch.split("_")[-1].replace("s.txt", ""))
      for i, linea in enumerate(archivo_abierto):
        datos = linea.split()

        x.append(float(datos[2]))
        y.append(float(datos[3]))
        z.append(float(datos[4]))
        t.append(i * step)
  return np.array(t), np.array(x), np.array(y), np.array(z)

def thomas_tridiagonal(a, b, c, d):
    """
    Resuelve un sistema de ecuaciones tridiagonal usando el algoritmo de Thomas.
    Versión optimizada que trabaja directamente con los elementos de la diagonal.
    
    Args:
        a (np.ndarray): Subdiagonal inferior (i, i-1)
        b (np.ndarray): Diagonal principal (i, i)
        c (np.ndarray): Superdiagonal superior (i, i+1)
        d (np.ndarray): Vector de términos independientes
    
    Returns:
        np.ndarray: Vector solución x
    """
    n = len(d)
    b_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    b_prime[0] = b[0]
    d_prime[0] = d[0]
    
    # Eliminación hacia adelante
    for i in range(1, n):
        m = a[i] / b_prime[i - 1]
        b_prime[i] = b[i] - m * c[i - 1]
        d_prime[i] = d[i] - m * d_prime[i - 1]
    
    # Sustitución hacia atrás
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1] / b_prime[n - 1]
    
    for i in range(n - 2, -1, -1):
        x[i] = (d_prime[i] - c[i] * x[i + 1]) / b_prime[i]
    
    return x

def calcular_coeficientes(y, h, c):
    """
    Calcula los coeficientes (a, b, c, d) del spline cúbico.
    
    Args:
        y (np.ndarray): Valores de las ordenadas
        h (np.ndarray): Diferencias entre puntos consecutivos del dominio
        c (np.ndarray): Coeficientes c del spline (segunda derivada)
    
    Returns:
        tuple: (a, b, c, d) - Coeficientes del spline cúbico en cada intervalo
    """
    n = len(y)

    a = y[:-1]

    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):

        b[i] = (
            (y[i + 1] - y[i]) / h[i]
            - h[i] * (2 * c[i] + c[i + 1]) / 3
        )

        d[i] = (
            (c[i + 1] - c[i])
            / (3 * h[i])
        )

    return a, b, c, d

def spline_natural(t, x):
    """
    Calcula los coeficientes de un spline cúbico natural.
    
    Args:
        t (np.ndarray): Puntos del dominio (tiempo)
        x (np.ndarray): Valores correspondientes a los puntos del dominio
    
    Returns:
        tuple: (t, (a, b, c, d)) - Puntos del dominio y coeficientes del spline
    """
    n = len(t)
    h = t[1:] - t[:-1]

    # Construir elementos de la matriz tridiagonal
    a_diag = np.zeros(n)  # subdiagonal inferior
    b_diag = np.zeros(n)  # diagonal principal
    c_diag = np.zeros(n)  # superdiagonal superior
    d_vec = np.zeros(n)   # vector de términos independientes

    # Condiciones naturales
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
    """
    Evalúa el spline cúbico de forma vectorizada en múltiples instantes de tiempo.
    Acepta un escalar o un array de tiempos. Usa búsqueda binaria para mejorar el rendimiento.

    Args:
        tiempos (np.ndarray o escalar): Instantes en los cuales se desea evaluar el spline
        t (np.ndarray): Puntos del dominio del spline
        a, b, c, d (np.ndarray): Coeficientes del spline cúbico

    Returns:
        np.ndarray o float: Valores del spline en los instantes especificados
    """
    # Soportar escalares convirtiéndolos en arrays temporales
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

def graficar_orbita(x, y, z):
    """
    Grafica una trayectoria tridimensional de una órbita.
    
    Args:
        x (np.ndarray): Coordenadas X de la órbita [km]
        y (np.ndarray): Coordenadas Y de la órbita [km]
        z (np.ndarray): Coordenadas Z de la órbita [km]
    """
    fig = plt.figure()

    ax = plt.axes(projection='3d')

    ax.scatter(x, y, z)

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")

    plt.show()

# Leer archivo
print("\n=== Inicializando lectura de datos ===")
print(f"Archivo: {ARCHIVO_1_SEG}")
t, x, y, z = leer_archivo(ARCHIVO_1_SEG)
print(f"✓ Datos cargados exitosamente")
print(f"  Número de registros: {len(t)}")
print(f"  Rango de tiempo: [{t[0]}, {t[-1]}] segundos")
print(f"  Rango X: [{x.min():.2f}, {x.max():.2f}] km")
print(f"  Rango Y: [{y.min():.2f}, {y.max():.2f}] km")
print(f"  Rango Z: [{z.min():.2f}, {z.max():.2f}] km")

# Aplicar función spline a cada coordenada
print("\n=== Calculando splines cúbicos naturales ===")
print("Procesando coordenada X...")
t_spline_x, coef_x = spline_natural(t, x)
print("✓ Spline X calculado")

print("Procesando coordenada Y...")
t_spline_y, coef_y = spline_natural(t, y)
print("✓ Spline Y calculado")

print("Procesando coordenada Z...")
t_spline_z, coef_z = spline_natural(t, z)
print("✓ Spline Z calculado")

# Evaluar el spline en puntos más finos para una mejor visualización
print("\n=== Evaluando splines en puntos interpolados ===")
t_fine = np.linspace(t[0], t[-1], min(len(t), 5000))
print(f"Interpolando en {len(t_fine)} puntos de tiempo...")
x_spline = evaluar_spline_vectorizado(t_fine, t_spline_x, *coef_x)
y_spline = evaluar_spline_vectorizado(t_fine, t_spline_y, *coef_y)
z_spline = evaluar_spline_vectorizado(t_fine, t_spline_z, *coef_z)
print("✓ Splines evaluados exitosamente")

print("\n=== Generando gráfica ===")
print("Las gráficas están disponibles en el menú interactivo.\n")

tiempos, pos_x, pos_y, pos_z = leer_archivo(ARCHIVO_600_SEG)

# nos quedamos solo con un período orbital (~110 min = 6600 s)
fin_periodo = 6600
indices_periodo = tiempos <= fin_periodo

tiempos_muestras = tiempos[indices_periodo]
x_muestras = pos_x[indices_periodo]
y_muestras = pos_y[indices_periodo]
z_muestras = pos_z[indices_periodo]

# construimos las splines para cada componente
t_x, (a_x, b_x, c_x, d_x) = spline_natural(tiempos_muestras, x_muestras)
t_y, (a_y, b_y, c_y, d_y) = spline_natural(tiempos_muestras, y_muestras)
t_z, (a_z, b_z, c_z, d_z) = spline_natural(tiempos_muestras, z_muestras)

# tiempos cada 60 segundos
tiempos_interpolados = np.arange(
    tiempos_muestras[0],
    tiempos_muestras[-1] + 60,
    60
)

# evaluamos las splines
x_interpolado = np.array([
    evaluar_spline_vectorizado(t_val, t_x, a_x, b_x, c_x, d_x)
    for t_val in tiempos_interpolados
])

y_interpolado = np.array([
    evaluar_spline_vectorizado(t_val, t_y, a_y, b_y, c_y, d_y)
    for t_val in tiempos_interpolados
])

z_interpolado = np.array([
    evaluar_spline_vectorizado(t_val, t_z, a_z, b_z, c_z, d_z)
    for t_val in tiempos_interpolados
])

"""Ecuación de posición: derivada de la posición = velocidad"""
def fr(r, v):
    return v

"""Ecuación de velocidad: derivada de la velocidad = aceleración gravitatoria"""
def fv(r):
    r_norm = np.linalg.norm(r)
    return -(MU / r_norm**3) * r

"""Calcula el estado (ri+1, vi+1) a partir del estado (ri, vi) usando RK4"""
def paso_rk4(ri, vi, h):

    # Primera evaluación
    k1r = fr(ri, vi)
    k1v = fv(ri)

    # Segunda evaluación
    k2r = fr(ri + 0.5 * h * k1r, vi + 0.5 * h * k1v)
    k2v = fv(ri + 0.5 * h * k1r)

    # Tercera evaluación
    k3r = fr(ri + 0.5 * h * k2r, vi + 0.5 * h * k2v)
    k3v = fv(ri + 0.5 * h * k2r)

    # Cuarta evaluación
    k4r = fr(ri + h * k3r, vi + h * k3v)
    k4v = fv(ri + h * k3r)

    # Actualización de la posición y la velocidad
    ri_1 = ri + (h / 6) * (k1r + 2*k2r + 2*k3r + k4r)
    vi_1 = vi + (h / 6) * (k1v + 2*k2v + 2*k3v + k4v)

    return ri_1, vi_1


"""Propaga la órbita desde el estado inicial durante tiempo_total segundos"""
def propagar_orbita(r0, v0, h, tiempo_total):
    n_pasos = int(tiempo_total / h)

    posiciones = np.zeros((n_pasos + 1, 3))
    velocidades = np.zeros((n_pasos + 1, 3))

    posiciones[0] = r0
    velocidades[0] = v0

    # Iteración temporal
    for i in range(n_pasos):
        posiciones[i+1], velocidades[i+1] = paso_rk4(posiciones[i], velocidades[i], h)

    return posiciones, velocidades

print("Trayectoria 3D disponible desde el menú interactivo.")

# Componentes X, Y y Z vs tiempo
print("\n=== Menú interactivo de gráficas ===")

def plot_full_orbit():
    graficar_orbita(x_spline, y_spline, z_spline)

def plot_period_3d():
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_interpolado, y_interpolado, z_interpolado, label="Spline interpolada")
    ax.scatter(x_muestras, y_muestras, z_muestras, color="red", s=60, label="Muestras (600 s)")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("Trayectoria orbital SAC-D — 1 período (~110 min)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figura_4_1_orbita_3D.png", dpi=150)
    plt.show()

def punto_c(spl_x, spl_y, spl_z, archivo_ref=ARCHIVO_1_SEG):
    """Comparar una spline construida con muestras (600s) contra la referencia 1s.

    spl_x/spl_y/spl_z: tuplas (t, a, b, c, d) con coeficientes del spline.
    archivo_ref: archivo con la referencia de 1s.
    """
    # Desempaquetar splines
    t_x, a_x, b_x, c_x, d_x = spl_x
    t_y, a_y, b_y, c_y, d_y = spl_y
    t_z, a_z, b_z, c_z, d_z = spl_z

    # Datos de referencia cada 1 segundo
    t_ref, x_ref, y_ref, z_ref = leer_archivo(archivo_ref)

    # Limitar la referencia al dominio de la spline
    t_max = t_x[-1]
    indices = t_ref <= t_max
    t_ref = t_ref[indices]
    x_ref = x_ref[indices]
    y_ref = y_ref[indices]
    z_ref = z_ref[indices]

    # Evaluar las splines en los mismos instantes
    x_spline = evaluar_spline_vectorizado(t_ref, t_x, a_x, b_x, c_x, d_x)
    y_spline = evaluar_spline_vectorizado(t_ref, t_y, a_y, b_y, c_y, d_y)
    z_spline = evaluar_spline_vectorizado(t_ref, t_z, a_z, b_z, c_z, d_z)

    # Diferencias
    dx = x_spline - x_ref
    dy = y_spline - y_ref
    dz = z_spline - z_ref

    # Errores máximos
    print("Error máximo X =", np.max(np.abs(dx)), "km")
    print("Error máximo Y =", np.max(np.abs(dy)), "km")
    print("Error máximo Z =", np.max(np.abs(dz)), "km")

    # Gráfico de las diferencias
    plt.figure(figsize=(10, 6))
    plt.plot(t_ref, dx, label="ΔX")
    plt.plot(t_ref, dy, label="ΔY")
    plt.plot(t_ref, dz, label="ΔZ")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error [km]")
    plt.title("Diferencias entre spline (600 s) y estimación precisa (1 s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    error_norma = np.sqrt(dx**2 + dy**2 + dz**2)
    print("Error máximo de posición =", np.max(error_norma), "km")

    print("\n=== Punto C: comparación spline(600s) vs referencia (1s) completa ===")
    print("Listo.")

def plot_components():
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(tiempos_interpolados, x_interpolado)
    axs[0].scatter(tiempos_muestras, x_muestras, color="red")
    axs[0].set_ylabel("X [km]")
    axs[0].grid()

    axs[1].plot(tiempos_interpolados, y_interpolado)
    axs[1].scatter(tiempos_muestras, y_muestras, color="red")
    axs[1].set_ylabel("Y [km]")
    axs[1].grid()

    axs[2].plot(tiempos_interpolados, z_interpolado)
    axs[2].scatter(tiempos_muestras, z_muestras, color="red")
    axs[2].set_ylabel("Z [km]")
    axs[2].set_xlabel("Tiempo [s]")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

def punto_a():
    """Simula y grafica tres órbitas de prueba (Punto A).

    Utiliza las funciones RK4 y propagar_orbita ya definidas para propagar
    tres órbitas de ejemplo y mostrar gráficos de trayectorias y conservación
    de energía.
    """
    # Parámetro gravitacional terrestre [km^3/s^2]
    MU_local = 398600.4418

    # CONDICIONES INICIALES
    P0_circular = np.array([6125.24, -3547.04, -3.31277])
    V0_circular = np.array([-0.519174, -0.903482, 7.43159])

    P0_geo = np.array([-41974.9, 4012.93, 23.5515])
    V0_geo = np.array([-0.292606, -3.06063, 0.000293508])

    P0_molniya = np.array([305.926, 3162.83, 6324.79])
    V0_molniya = np.array([-9.86976, 0.326984, 0.313879])

    # Periodos aproximados de cada órbita en segundos
    T_circular = 6000
    T_geo      = 86400
    T_molniya  = 43200

    h_local = 1

    print("Propagando órbita circular...")
    pos_circ, vel_circ = propagar_orbita(P0_circular, V0_circular, h_local, T_circular)

    print("Propagando órbita geoestacionaria...")
    pos_geo, vel_geo = propagar_orbita(P0_geo, V0_geo, h_local, T_geo)

    print("Propagando órbita Molniya...")
    pos_mol, vel_mol = propagar_orbita(P0_molniya, V0_molniya, h_local, T_molniya)

    # Gráfico de las órbitas en 3D
    fig = plt.figure(figsize=(10, 8))
    ax_local = fig.add_subplot(111, projection='3d')

    ax_local.plot(pos_circ[:, 0], pos_circ[:, 1], pos_circ[:, 2], label='Circular')
    ax_local.plot(pos_geo[:, 0],  pos_geo[:, 1],  pos_geo[:, 2],  label='Geoestacionaria')
    ax_local.plot(pos_mol[:, 0],  pos_mol[:, 1],  pos_mol[:, 2],  label='Molniya')

    ax_local.scatter(*P0_circular, s=50, zorder=5, color='blue')
    ax_local.scatter(*P0_geo,      s=50, zorder=5, color='orange')
    ax_local.scatter(*P0_molniya,  s=50, zorder=5, color='green')

    ax_local.set_xlabel('X [km]')
    ax_local.set_ylabel('Y [km]')
    ax_local.set_zlabel('Z [km]')
    ax_local.set_title('Órbitas propagadas con RK4')
    ax_local.legend()
    plt.tight_layout()
    plt.savefig('figura_orbitas_3D.png', dpi=150)

    # Conservación de la energía para las tres órbitas
    orbitas = [
        ("Circular", pos_circ, vel_circ),
        ("Geoestacionaria", pos_geo, vel_geo),
        ("Molniya", pos_mol, vel_mol)
    ]

    plt.figure(figsize=(10,6))

    for nombre, pos, vel in orbitas:
        energia = (0.5 * np.sum(vel**2, axis=1)) - MU_local / np.linalg.norm(pos, axis=1)
        error_rel = np.abs((energia - energia[0]) / energia[0])
        print(f"{nombre}: Error relativo máximo = {np.max(error_rel):.3e}")
        plt.plot(error_rel, label=nombre)

    plt.xlabel("Paso temporal")
    plt.ylabel("Error relativo de energía")
    plt.title("Conservación de energía con RK4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    r0 = np.linalg.norm(pos_circ[0])
    rf = np.linalg.norm(pos_circ[-1])
    print("Radio inicial =", r0)
    print("Radio final   =", rf)
    print("Diferencia    =", rf-r0)

    # Dibujar superficie terrestre como contexto
    u = np.linspace(0,2*np.pi,50)
    v = np.linspace(0,np.pi,50)
    R = 6378
    x = R*np.outer(np.cos(u),np.sin(v))
    y = R*np.outer(np.sin(u),np.sin(v))
    z = R*np.outer(np.ones(np.size(u)),np.cos(v))
    ax_local.plot_surface(x,y,z,alpha=0.3)
    plt.show()


def menu():
    while True:
        print("\nElija qué graficar:")
        print("1) Órbita completa 3D (interpolada)")
        print("2) Órbita 1 período 3D (muestras + spline)")
        print("3) Componentes X/Y/Z vs tiempo")
        print("4) Punto C: comparar spline(600s) vs referencia(1s)")
        print("5) Punto A: simular órbitas RK4")
        print("6) Salir")
        opt = input("Opción [1-6]: ").strip()
        if opt == "1":
            plot_full_orbit()
        elif opt == "2":
            plot_period_3d()
        elif opt == "3":
            plot_components()
        elif opt == "4":
            spl_x = (t_x, a_x, b_x, c_x, d_x)
            spl_y = (t_y, a_y, b_y, c_y, d_y)
            spl_z = (t_z, a_z, b_z, c_z, d_z)
            punto_c(spl_x, spl_y, spl_z)
        elif opt == "5":
            punto_a()
        elif opt == "6":
            print("Saliendo menú.")
            break
        else:
            print("Opción inválida, intente otra vez.")

menu()