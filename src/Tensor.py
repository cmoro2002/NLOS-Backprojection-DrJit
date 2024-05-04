import drjit as dr

from drjit.llvm import Float, Int, Array3f, UInt32, Loop, UInt, TensorXf

# Clase Tensor2f que permite la lecutra de un tensor de dos dimensiones
# como si lo permitiese DrJit

# Shape define el tamaño del tensor de la forma [altura, ancho]
class Tensor2f:
    def __init__(self, data: Float, shape: tuple):
        self.data = data
        self.height = shape[0]
        self.width = shape[1]

    def leer(self, y: Int, x: Int) -> Float:
        ys = y * self.width

        indices = ys + x
        return dr.gather(Float, self.data, indices)
    
    def calcularIndices(self, y: Int, x: Int) -> Int:
        ys = y * self.width
        return ys + x
    
    def leerIndice(self, indices: Int) -> Float:
        return dr.gather(Float, self.data, indices)
