import numpy as np
import drjit as dr

# from drjit.cuda import Float, UInt32
from drjit.llvm import Float, Int, Array3f, Matrix3f, TensorXf

def calcular_posicion_aplanada(coordenadas, forma):
    """
    Calcula la posición aplanada en un tensor dado un conjunto de coordenadas tridimensionales.

    Parámetros:
    - coordenadas: Tupla de tres elementos que representan las coordenadas tridimensionales.
    - forma: Tupla de tres elementos que representan la forma del tensor tridimensional.

    Retorna:
    - La posición aplanada correspondiente a las coordenadas dadas.
    """
    i, j, k = coordenadas
    M, N, P = forma
    return i * N * P + j * P + k
class TransientImage:
    def __init__(self, width: int, height: int, channels: int, time_per_coord: float, intensity_multiplier_unit: float, data: TensorXf, max_value: float, min_value: float):
        self.width = width
        self.height = height
        self.channels = channels
        self.time_per_coord = time_per_coord
        self.intensity_multiplier_unit = intensity_multiplier_unit
        if data is not None:
            self.data = Float(data.array)
        else:
            self.data = dr.zeros(Array3f, shape=(height, width, channels))
        self.maxValue = max_value
        self.minValue = min_value
        self.laserHitTime = 0
        self.wallCameraDilation = np.zeros(height)
        self.point_wall_i = dr.zeros(Array3f,1)
        self.wallDirection = dr.zeros(Array3f,1)
        self.wallNormal = dr.zeros(Array3f,1)
        self.laser = dr.zeros(Array3f,1)
        self.wallViewWidth = height
        self.pxHalfWidth = 0.5

    # Devuelve la intensidad de la imagen en el punto (x, y)
    def getIntensityForTime(self, y: int, time: float) -> float:
        x = int((time + self.laserHitTime + self.wallCameraDilation[y]) / self.time_per_coord)
        if x >= self.width or x < 0:
            return 0
        
        # return self.data[x, y, 0]
        return self.data[calcular_posicion_aplanada((x, y, 0), (self.width, self.height, self.channels))]
    
    # Devuelve el punto correspondiente a la coordenada y
    def getPointForCoord(self, y: Int, aux=None):
        if aux is None:
            aux = dr.zeros(Array3f, 1)  # Inicializa un vector auxiliar si no se proporciona uno

        aux[0] = self.point_wall_i[0] + (((float(y) / self.height) * self.wallViewWidth + self.pxHalfWidth) * self.wallDirection[0])
        aux[1] = self.point_wall_i[1] + (((float(y) / self.height) * self.wallViewWidth + self.pxHalfWidth) * self.wallDirection[1])
        aux[2] = self.point_wall_i[2] + (((float(y) / self.height) * self.wallViewWidth + self.pxHalfWidth) * self.wallDirection[2])
        
        return aux
    
    # Método para obtener el valor del atributo laser
    def get_laser(self):
        return self.laser
    
    def setLaserHitTime(self, t):
        self.laserHitTime = t

    def getLaser(self):
        return self.laser

    def __str__(self):
        return f"TransientImage: width={self.width}, height={self.height}, channels={self.channels}, time_per_coord={self.time_per_coord}, intensity_multiplier_unit={self.intensity_multiplier_unit}, max_value={self.maxValue}, min_value={self.minValue}"
