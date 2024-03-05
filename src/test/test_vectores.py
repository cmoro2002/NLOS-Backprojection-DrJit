import numpy as np
import time

# Definir la dimensión del vector
dimension = 3000000000

# Definir las matrices para las operaciones
matriz_A = np.random.rand(dimension)
matriz_B = np.random.rand(dimension)

# Función para medir el tiempo de ejecución de cada operación
def medir_tiempo(operacion, *args):
    inicio = time.perf_counter()
    print("Inicio:", inicio)
    resultado = operacion(*args)
    fin = time.perf_counter()
    print("Fin:", fin)
    return fin - inicio

# Función para la multiplicación de matrices
def multiplicacion(matriz, vector):
    return matriz @ vector

# Función para la suma de matrices
def suma(matriz_A, matriz_B):
    return (matriz_A + matriz_B)

def sumar_vector(matriz_A):
    suma = np.sum(matriz_A)
    print("La suma de todas las componentes del vector es:", suma)
    return suma

# Calcular el dot product de dos vectores
def multiplicar_vector(matriz_A, matriz_B):
    return np.dot(matriz_A, matriz_B)

# Medir el tiempo para la multiplicación de matrices
# tiempo_multiplicacion = medir_tiempo(multiplicacion, matriz_A, vector)

# Medir el tiempo para la suma de matrices
# tiempo_suma = medir_tiempo(suma, matriz_A, matriz_B)

# Medir el tiempo para la suma de un vector
tiempo_suma_vector = medir_tiempo(sumar_vector, matriz_A)

# Medir el tiempo para la multiplicación de un vector
tiempo_multiplicacion_vector = medir_tiempo(multiplicar_vector, matriz_A, matriz_B)

# Imprimir los resultados
# print(f"Tiempo de multiplicación: {tiempo_multiplicacion:.5f} segundos")
# print(f"Tiempo de suma: {tiempo_suma:.5f} segundos")
# print(f"Tiempo de suma de vector: {tiempo_suma_vector:.5f} segundos")
print(f"Tiempo de multiplicación de vector: {tiempo_multiplicacion_vector:.5f} segundos")
