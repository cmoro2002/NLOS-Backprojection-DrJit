import numpy as np
import threading
import time

import drjit as dr

from drjit.llvm import Float, Int, Array3f, UInt32, Loop, UInt, Array2f

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def foo(self):
        return self.x * self.y

# Creamos algunas instancias de Point
points = [Point(1, 2), Point(3, 4), Point(5, 6)]

# Supongamos que tenemos una matriz de punteros a estas instancias (esta parte depende de cómo se implementa el registro Dr.Jit)
pointers = Array2f(points)

# Definimos la función que queremos despachar
def func(self, arg):
    v = self.foo()  # Llama al método foo() en cada instancia
    return v + arg

# Supongamos que tenemos un argumento para pasar a la función
arg = 10

# Llamamos a drjit.dispatch() para ejecutar la función en cada instancia de Point
result = dr.dispatch(pointers, func, arg)

print(result)
