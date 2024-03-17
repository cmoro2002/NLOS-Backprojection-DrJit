import numpy as np
import drjit as dr
import time

from drjit.llvm import Float, Int, Array3f, Matrix3f, Loop, UInt32

# Crear un vector aleatorio
dimension = 50000000
vector = np.random.rand(dimension)

# Crear un vector en drjit
vector_drjit = Float(vector)
i = UInt32(0)
sumatorio = Float(0)

# Definir el loop que se utilizará para la suma
loop = Loop("Sumar", lambda: (sumatorio, i))

# Medir cuanto tarda en sumar el vector
inicio = time.time()

# Definir el cuerpo del loop
while loop(i < dimension):
    sumatorio += dr.gather(Float, vector_drjit, i)
    i += 1

print("La suma de todas las componentes del vector es:", sumatorio)

# Medir el tiempo que tarda en sumar el vector
fin = time.time()

# Imprimir el resultado
print(f"Tiempo de suma de vector: {fin - inicio:.5f} segundos")

### Prueba con fors normales ### 

# Medir cuanto tarda en sumar el vector
inicio = time.time()

sumatorio = 0

# Definir el cuerpo del loop
for i in range(dimension):
    sumatorio += vector[i]

# Imprimir el resultado
print("La suma de todas las componentes del vector es:", sumatorio)

# Medir el tiempo que tarda en sumar el vector
fin = time.time()
print(f"Tiempo de suma de vector: {fin - inicio:.5f} segundos")

print("Resultados reales (numpy): ", np.sum(vector))
print("Resultados reales (drjit): ", dr.sum(vector_drjit))
