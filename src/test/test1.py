import numpy as np

def suma_matrices(a, b):
  """
  Suma dos matrices de la misma forma.

  Parámetros:
    a (np.ndarray): Matriz de entrada.
    b (np.ndarray): Matriz de entrada.

  Retorno:
    np.ndarray: Matriz con la suma de las entradas de `a` y `b`.
  """
  c = np.zeros_like(a)
  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      c[i, j] = a[i, j] + b[i, j]
  return c

import time

a = np.random.rand(5000, 5000)
b = np.random.rand(5000, 5000)

start = time.time()
c = suma_matrices(a, b)
end = time.time()

tiempo_sin_gpu = end - start

print(f"Tiempo sin GPU: {tiempo_sin_gpu}")

from numba import jit

@jit
def suma_matrices_jit(a, b):
  """
  Suma dos matrices de la misma forma.

  Parámetros:
    a (np.ndarray): Matriz de entrada.
    b (np.ndarray): Matriz de entrada.

  Retorno:
    np.ndarray: Matriz con la suma de las entradas de `a` y `b`.
  """
  c = np.zeros_like(a)
  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      c[i, j] = a[i, j] + b[i, j]
  return c

a = np.random.rand(5000, 5000)
b = np.random.rand(5000, 5000)

start = time.time()
c = suma_matrices_jit(a, b)
end = time.time()

tiempo_con_gpu = end - start

print(f"Tiempo con GPU: {tiempo_con_gpu}")

