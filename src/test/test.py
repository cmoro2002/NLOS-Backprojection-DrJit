import numpy as np
import threading
import time

import drjit as dr

def matriz_multiplicacion(matriz_a, matriz_b, resultado, inicio_fila, fin_fila):
    for i in range(inicio_fila, fin_fila):
        for j in range(matriz_b.shape[1]):
            resultado[i][j] = np.sum(matriz_a[i,:] * matriz_b[:,j])

def multiplicacion_sin_hilos(matriz_a, matriz_b):
    resultado = np.zeros((matriz_a.shape[0], matriz_b.shape[1]))
    matriz_multiplicacion(matriz_a, matriz_b, resultado, 0, matriz_a.shape[0])
    return resultado

def multiplicacion_con_hilos(matriz_a, matriz_b, num_hilos):
    num_filas_por_hilo = matriz_a.shape[0] // num_hilos
    hilos = []
    resultado = np.zeros((matriz_a.shape[0], matriz_b.shape[1]))

    for i in range(num_hilos):
        inicio_fila = i * num_filas_por_hilo
        fin_fila = inicio_fila + num_filas_por_hilo
        if i == num_hilos - 1:
            fin_fila = matriz_a.shape[0]
        hilo = threading.Thread(target=matriz_multiplicacion, args=(matriz_a, matriz_b, resultado, inicio_fila, fin_fila))
        hilos.append(hilo)
        hilo.start()

    for hilo in hilos:
        hilo.join()

    return resultado

# Definir el tamaño de las matrices
tamano_matriz = 3000

# Generar matrices aleatorias
matriz_a = np.random.rand(tamano_matriz, tamano_matriz)
matriz_b = np.random.rand(tamano_matriz, tamano_matriz)

# Multiplicación sin hilos
inicio_sin_hilos = time.time()
resultado_sin_hilos = multiplicacion_sin_hilos(matriz_a, matriz_b)
fin_sin_hilos = time.time()
tiempo_sin_hilos = fin_sin_hilos - inicio_sin_hilos
print("Tiempo sin hilos:", tiempo_sin_hilos)

# Multiplicación con hilos (2 hilos)
num_hilos = 8
inicio_con_hilos = time.time()
resultado_con_hilos = multiplicacion_con_hilos(matriz_a, matriz_b, num_hilos)
fin_con_hilos = time.time()
tiempo_con_hilos = fin_con_hilos - inicio_con_hilos
print("Tiempo con", num_hilos, "hilo(s):", tiempo_con_hilos)
