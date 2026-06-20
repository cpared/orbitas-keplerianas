## Requisitos

- Python 3.7+
- NumPy
- Matplotlib

## Instalación de dependencias

```bash
pip install numpy matplotlib
```

## Ejecución del programa

```bash
python3 main.py
```

El programa cargará los datos de los archivos de entrada y presentará un menú interactivo con 6 opciones.

Si bien, cargar los archivos en memoria no es recomendable debido a su tamaño, lo hacemos para facilitar
la carga del menu.

## Menú principal

### Opción 1: Órbita completa 3D (interpolada)

Grafica la órbita SAC-D completa interpolada mediante splines cúbicos naturales en 3D.

**Qué hace:**
- Lee el archivo de datos de 1 segundo (SACD_TPV_step_1s.txt)
- Calcula splines cúbicos naturales para cada coordenada (X, Y, Z)
- Interpola los datos en 5000 puntos suavizados
- Muestra una gráfica 3D de dispersión de la órbita completa

**Salida:** Gráfica interactiva en 3D

---

### Opción 2: Órbita 1 período 3D (muestras + spline)

Grafica la órbita durante un período orbital (~110 minutos), mostrando simultáneamente las muestras originales y la interpolación.

**Qué hace:**
- Lee datos con muestreo cada 600 segundos
- Limita al primer período orbital (6600 segundos)
- Calcula splines a partir de las muestras
- Grafica en 3D:
  - **Línea azul:** Interpolación por spline
  - **Puntos rojos:** Muestras originales cada 600s
- Guarda la figura como `figura_4_1_orbita_3D.png`

**Salida:** Gráfica 3D + imagen PNG

---

### Opción 3: Componentes X/Y/Z vs tiempo

Grafica por separado las tres componentes cartesianas como función del tiempo.

**Qué hace:**
- Extrae las posiciones interpoladas del período orbital
- Crea 3 subgráficas:
  - **Gráfica superior:** Componente X vs tiempo (línea azul + puntos rojos de muestras)
  - **Gráfica central:** Componente Y vs tiempo
  - **Gráfica inferior:** Componente Z vs tiempo
- Todos los gráficos comparten el eje temporal para facilitar la comparación

**Salida:** Gráfica de 3 subpaneles

---

### Opción 4: Punto C - Comparar spline(600s) vs referencia(1s)

Análisis de errores comparando la interpolación por splines (construida con muestras cada 600s) contra la referencia de alta precisión (cada 1 segundo).

**Qué hace:**
- Carga los datos de referencia cada 1 segundo
- Carga los datos de muestreo cada 600 segundos y construye splines
- Evalúa ambas series en los mismos instantes de tiempo
- Calcula las diferencias en X, Y, Z
- Imprime en consola:
  - Error máximo en X
  - Error máximo en Y
  - Error máximo en Z
  - Error máximo de posición (norma euclidiana)
- Muestra gráfica con tres líneas:
  - ΔX: diferencia en componente X
  - ΔY: diferencia en componente Y
  - ΔZ: diferencia en componente Z

**Salida:** Impresión de errores + gráfica temporal de diferencias

---

### Opción 5: Punto A - Simulación de órbitas RK4

Simula la propagación de tres órbitas de prueba (circular, geoestacionaria y Molniya) usando integración Runge-Kutta de orden 4.

**Qué hace:**
- Define tres órbitas con condiciones iniciales predeterminadas:
  - **Circular:** Órbita LEO (Low Earth Orbit)
  - **Geoestacionaria:** Órbita GEO (Geosynchronous)
  - **Molniya:** Órbita excéntrica de comunicaciones
- Propaga cada órbita usando RK4 con paso de 1 segundo
- Calcula y valida la conservación de energía
- Imprime en consola:
  - Error relativo de energía para cada órbita
  - Radio inicial y final de la órbita circular (validación)
  - Diferencia de radios
- Muestra gráficas:
  1. **Órbitas en 3D:** Las tres trayectorias con la Tierra como referencia (esfera)
  2. **Conservación de energía:** Error relativo de energía vs tiempo para cada órbita

**Salida:** Dos gráficas + impresión de métricas

---

### Opción 6: Salir

Cierra el programa.

---

## Estructura de archivos

```
TP2-orbitas-keplerianas/
├── main.py              # Menú principal y coordinación
├── data.py              # Lectura de archivos
├── spline.py            # Cálculo de splines cúbicos
├── rk4.py               # Integración RK4
├── calculos.py          # Orquestación de cálculos
├── plots.py             # Funciones de graficación
├── README.md            # Este archivo
├── SACD_TPV_step_1s.txt     # Datos de entrada (1 seg)
├── SACD_TPV_step_600s.txt   # Datos de entrada (600 seg)
└── figura_*.png         # Gráficas generadas
```

---

## Módulos

### `data.py`
Lectura de archivos de órbita con formato: FECHA HORA POS_X POS_Y POS_Z VEL_X VEL_Y VEL_Z

### `spline.py`
Implementación de splines cúbicos naturales:
- Algoritmo de Thomas para sistemas tridiagonales
- Cálculo de coeficientes de Hermite
- Evaluación vectorizada de splines

### `rk4.py`
Integración Runge-Kutta de orden 4 para dinámicas orbitales:
- Ecuaciones diferenciales de movimiento orbital
- Cálculo de energía
- Análisis de errores

### `calculos.py`
Funciones de orquestación que combinan módulos:
- Interpolación de órbitas con splines
- Análisis comparativo de errores
- Simulación de órbitas de prueba

### `plots.py`
Funciones de graficación independientes:
- Órbitas 3D
- Series temporales
- Análisis de errores
- Conservación de energía

### `main.py`
Menú interactivo que coordina todas las funciones.

---

## Notas

- Las gráficas se generan en ventanas interactivas (requiere entorno gráfico)
- Los archivos PNG se guardan en el directorio de ejecución
- Puede requerirse presionar Enter o cerrar ventanas gráficas para continuar el menú
- Los tiempos de cálculo dependen del tamaño de los datos (típicamente segundos a minutos)

---

## Autor

Trabajo práctico de Modelación Numérica - FIUBA
