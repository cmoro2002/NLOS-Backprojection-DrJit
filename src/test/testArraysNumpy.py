import numpy as np
import time

# Función para la multiplicación de matrices
def multiplicacion(matriz, vector):
    return matriz.dot(vector)

# Función para la suma de matrices
def suma(matriz_A, matriz_B):
    return matriz_A + matriz_B

def sumar_vector(vector):
    suma = np.sum(vector)
    print("La suma de todas las componentes del vector es:", suma)
    return suma

def medir_tiempo(operacion, *args):
    inicio = time.time()
    resultado = operacion(*args)
    print(resultado)
    fin = time.time()
    return round(fin - inicio,3)

# Definir las dimensiones iniciales y finales
dimension_inicial = 100000
dimension_final = 500000000
multiplicador = 2

# Archivo para escribir los resultados
nombre_archivo = "resultados/Numpy.txt"

with open(nombre_archivo, "w") as archivo:
    archivo.write("dimension tiempo_multiplicacion tiempo_suma tiempo_suma_vector\n")

    # Iterar sobre diferentes dimensiones
    dimension_actual = dimension_inicial
    while dimension_actual <= dimension_final:
        # Crear un vector aleatorio
        vector = np.random.rand(dimension_actual)

        # Definir las matrices para las operaciones
        matriz_A = np.random.rand(dimension_actual)
        matriz_B = np.random.rand(dimension_actual)

        # Medir el tiempo para la multiplicación de matrices
        tiempo_multiplicacion = medir_tiempo(multiplicacion, matriz_A, vector)

        # Medir el tiempo para la suma de matrices
        tiempo_suma = medir_tiempo(suma, matriz_A, matriz_B)

        # Medir el tiempo que tarda en sumar_vector
        tiempo_suma_vector = medir_tiempo(sumar_vector, matriz_A)

        # Escribir los resultados en el archivo
        archivo.write(f"{dimension_actual} {tiempo_multiplicacion} {tiempo_suma} {tiempo_suma_vector}\n")

        # Incrementar la dimensión para la próxima iteración
        dimension_actual *= multiplicador
