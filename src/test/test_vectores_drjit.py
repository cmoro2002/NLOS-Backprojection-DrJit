import numpy as np
import drjit as dr
import time

from drjit.llvm import Float, Int, Array3f, Matrix3f


# Definir la dimensión del vector
dimension = 300000000

# Crear un vector aleatorio
vector = np.random.rand(dimension)

# Definir las matrices para las operaciones
matriz_A = np.random.rand(dimension)
matriz_B = np.random.rand(dimension)

matriz_C = Float(matriz_A)

print("El tipo de las matrices es:", type(matriz_A), type(matriz_B), type(matriz_C))

# Función para medir el tiempo de ejecución de cada operación
def medir_tiempo(operacion, *args):
    inicio = time.time()
    resultado = operacion(*args)
    fin = time.time()
    print(resultado)
    return fin - inicio

# Función para la multiplicación de matrices
def multiplicacion(matriz, vector):
    return matriz @ vector

# Función para la suma de matrices
def suma(matriz_A, matriz_B):
    return (matriz_A + matriz_B)

def sumar_vector(vector):
    suma = dr.sum(vector)
    print("La suma de todas las componentes del vector es:", suma)
    return suma

# Medir el tiempo para la multiplicación de matrices
# tiempo_multiplicacion = medir_tiempo(multiplicacion, matriz_A, vector)

# Medir el tiempo para la suma de matrices
# tiempo_suma = medir_tiempo(suma, matriz_A, matriz_B)

# Medir el tiempo que tarda en sumar_vector
tiempo_suma_vector = medir_tiempo(sumar_vector, matriz_C)

# Imprimir los resultados
# print(f"Tiempo de multiplicación: {tiempo_multiplicacion:.5f} segundos")
# print(f"Tiempo de suma: {tiempo_suma:.5f} segundos")
print(f"Tiempo de suma de vector: {tiempo_suma_vector:.5f} segundos")

