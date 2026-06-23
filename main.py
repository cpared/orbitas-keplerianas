"""Main - Menú interactivo de visualización de órbitas."""
import numpy as np
from data import leer_archivo
from calculos import calcular_orbita_spline, calcular_error_spline, calcular_orbitas_rk4, calcular_magnitudes_24h
from plots import (
    plot_orbita_3d, plot_orbita_3d_con_spline, plot_componentes_vs_tiempo,
    plot_error_spline, plot_conservacion_energia, plot_orbitas_3d_multiples,
    plot_magnitudes_orbitales
)


# Configuración
ARCHIVO_1_SEG = "SACD_TPV_step_1s.txt"
ARCHIVO_600_SEG = "SACD_TPV_step_600s.txt"

# Condiciones iniciales para Punto A
P0_circular = np.array([6125.24, -3547.04, -3.31277])
V0_circular = np.array([-0.519174, -0.903482, 7.43159])

P0_geo = np.array([-41974.9, 4012.93, 23.5515])
V0_geo = np.array([-0.292606, -3.06063, 0.000293508])

P0_molniya = np.array([305.926, 3162.83, 6324.79])
V0_molniya = np.array([-9.86976, 0.326984, 0.313879])


# Variables globales (cacheadas al iniciar)
datos_spline = None
datos_rk4 = None


def inicializar():
    """Carga datos necesarios una sola vez."""
    global datos_spline, datos_rk4
    print("Inicializando datos...")
    datos_spline = calcular_orbita_spline(ARCHIVO_1_SEG, ARCHIVO_600_SEG)
    
    condiciones = [
        ("Circular", P0_circular, V0_circular, 6000),
        ("Geoestacionaria", P0_geo, V0_geo, 86400),
        ("Molniya", P0_molniya, V0_molniya, 43200)
    ]
    datos_rk4 = calcular_orbitas_rk4(condiciones)
    print("✓ Datos cargados")


def menu_opcion_1():
    """Opción 1: Órbita completa 3D interpolada."""
    _, x, y, z = datos_spline['orbita_completa']
    plot_orbita_3d(x, y, z, "Órbita completa interpolada (1s)")


def menu_opcion_2():
    """Opción 2: Órbita período 3D con spline y muestras."""
    t_interp, x_interp, y_interp, z_interp, t_muestras, x_muestras, y_muestras, z_muestras = datos_spline['orbita_periodo']
    plot_orbita_3d_con_spline(x_interp, y_interp, z_interp, x_muestras, y_muestras, z_muestras,
                              "Trayectoria orbital SAC-D — 1 período (~110 min)",
                              "figura_4_1_orbita_3D.png")


def menu_opcion_3():
    """Opción 3: Componentes X/Y/Z vs tiempo."""
    t_interp, x_interp, y_interp, z_interp, t_muestras, x_muestras, y_muestras, z_muestras = datos_spline['componentes']
    plot_componentes_vs_tiempo(t_interp, x_interp, y_interp, z_interp, t_muestras, x_muestras, y_muestras, z_muestras)


def menu_opcion_4():
    """Opción 4: Comparar spline(600s) vs referencia(1s)."""
    t_x_m, coef_x_m, t_y_m, coef_y_m, t_z_m, coef_z_m = datos_spline['splines']
    spl_x = (t_x_m, *coef_x_m)
    spl_y = (t_y_m, *coef_y_m)
    spl_z = (t_z_m, *coef_z_m)
    
    errores = calcular_error_spline(spl_x, spl_y, spl_z, ARCHIVO_1_SEG, t_x_m[-1])
    
    print(f"\nError máximo X = {errores['error_max_x']:.2f} km")
    print(f"Error máximo Y = {errores['error_max_y']:.2f} km")
    print(f"Error máximo Z = {errores['error_max_z']:.2f} km")
    print(f"Error máximo de posición = {errores['error_max_posicion']:.2f} km")
    
    plot_error_spline(errores['tiempos'], errores['dx'], errores['dy'], errores['dz'])


def menu_opcion_5():
    """Opción 5: Simulación RK4 de órbitas."""
    posiciones = [d['posiciones'] for d in datos_rk4]
    nombres = [d['nombre'] for d in datos_rk4]
    pos_iniciales = [P0_circular, P0_geo, P0_molniya]
    
    print("\nErrores de energía:")
    for d in datos_rk4:
        print(f"{d['nombre']}: {np.max(d['error_energia']):.3e}")
    
    # Radio inicial vs final (circular)
    r0 = np.linalg.norm(posiciones[0][0])
    rf = np.linalg.norm(posiciones[0][-1])
    print(f"\nRadio inicial circular = {r0:.2f} km")
    print(f"Radio final circular   = {rf:.2f} km")
    print(f"Diferencia             = {rf - r0:.2f} km")
    
    # Gráficos
    plot_orbitas_3d_multiples(posiciones, nombres, pos_iniciales, "Órbitas propagadas con RK4", "figura_orbitas_3D.png")
    
    errores_energia = [d['error_energia'] for d in datos_rk4]
    plot_conservacion_energia(errores_energia, nombres)


def menu_opcion_6():
    """Opción 6: Punto C (opcional) — Momento angular y energía (24h)."""
    print("\nPropagando órbitas durante 24h...")
    condiciones = [
        ("Circular", P0_circular, V0_circular, 86400),
        ("Geoestacionaria", P0_geo, V0_geo, 86400),
        ("Molniya", P0_molniya, V0_molniya, 86400)
    ]
    resultados = calcular_magnitudes_24h(condiciones, h=1, tiempo_total=86400)
    print("✓ Propagación completada")
    plot_magnitudes_orbitales(resultados)


def menu():
    """Menú interactivo principal."""
    while True:
        print("\n" + "="*50)
        print("Seleccione una opción:")
        print("1) Órbita completa 3D (interpolada)")
        print("2) Órbita 1 período 3D (muestras + spline)")
        print("3) Componentes X/Y/Z vs tiempo")
        print("4) Punto C: comparar spline(600s) vs referencia(1s)")
        print("5) Punto A: simular órbitas RK4")
        print("6) Punto C (opcional): Momento angular y energía (24h)")
        print("7) Salir")
        print("="*50)
        
        opt = input("Opción [1-7]: ").strip()
        
        if opt == "1":
            menu_opcion_1()
        elif opt == "2":
            menu_opcion_2()
        elif opt == "3":
            menu_opcion_3()
        elif opt == "4":
            menu_opcion_4()
        elif opt == "5":
            menu_opcion_5()
        elif opt == "6":
            menu_opcion_6()
        elif opt == "7":
            print("Saliendo.")
            break
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    inicializar()
    menu()
