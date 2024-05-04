import numpy as np
import drjit as dr

# from drjit.cuda import Float, UInt32
from drjit.llvm import Float, Int, Array3f, Matrix3f, TensorXf
from Tensor import Tensor2f

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

def calcular_posiciones_aplanadas(x: np.ndarray, y: np.ndarray, forma: tuple) -> np.ndarray:
    """
    Calcula las posiciones aplanadas en un tensor dado un conjunto de coordenadas bidimensionales.

    Parámetros:
    - x: Vector de coordenadas en el eje x.
    - y: Vector de coordenadas en el eje y.
    - forma: Tupla de tres elementos que representan la forma del tensor tridimensional.

    Retorna:
    - Un vector con las posiciones aplanadas correspondientes a las coordenadas dadas.
    """
    M, N, P = forma
    return x * N * P + y * P

class TransientImage:
    def __init__(self, width: Int, height: Int, channels: Int, time_per_coord: Float, intensity_multiplier_unit: Float, data: np.ndarray, max_value: Float, min_value: Float):
        self.width = width
        self.height = height
        self.channels = channels
        self.time_per_coord = time_per_coord
        self.intensity_multiplier_unit = intensity_multiplier_unit
        if data is not None:
            self.data = data
        # else:
            # self.data = dr.zeros(Array3f, shape=(height, width, channels))
        tensorAux = TensorXf(self.data)
        self.tensor = Tensor2f(tensorAux.array, (height, width))
        self.maxValue = max_value
        self.minValue = min_value
        self.laserHitTime = 0
        # self.wallCameraDilation = dr.zeros(Float,height)
        self.wallCameraDilation = np.zeros(height, dtype=float)
        # self.point_wall_i = dr.zeros(Array3f,1)
        self.point_wall_i = np.zeros(3, dtype=float)
        self.wallDirection = dr.zeros(Array3f,1)
        self.wallNormal = dr.zeros(Array3f,1)
        # self.laser = dr.zeros(Array3f,1)
        self.laser = np.zeros(3, dtype=float)
        self.wallViewWidth = height
        self.pxHalfWidth = 0.5
        self.wallPoints = None

    def setWallPoints(self, wallPoints):
        self.wallPoints = wallPoints

    # Devuelve la intensidad de la imagen en el punto (x, y)
    def getIntensityForTime(self, y: int, time: float) -> float:
        x = int((time + self.laserHitTime + self.wallCameraDilation[y])  / self.time_per_coord)
        if x >= self.width or x < 0:
            return 0
        
        # print(f"Índice de la coordenada x: {x}")
        
        return self.data[y, x, 0]
        # return self.data[calcular_posicion_aplanada((x, y, 0), (self.width, self.height, self.channels))]
    
    # Devuelve el tiempo correspondiente a las coordenadas y sus tiempos
    def getIntensitiesForTime(self, y: np.ndarray, times: Float) -> Float:
        # Calcular los índices de las coordenadas x para todos los tiempos
        # TODO: wallCameraDilatation sea de drjit y acceder con scatter
        x = Int(((times + self.laserHitTime + self.wallCameraDilation[y]) / self.time_per_coord))

        # Filtrar los índices que están fuera de los límites de la imagen
        x = dr.clip(x, 0, self.width - 1)
        # print(self.data[y, x, 0])

        # Mostrar los valores de x e y
        return self.tensor.leer(y, x)

    # dr.fma(offset, self.wallDirection, self.point_wall_i)

    # Devuelve el punto correspondiente a la coordenada y
    def getPointForCoord(self, y: int, aux=None):     
        if aux is None:
            aux = np.zeros(3, dtype=float)
            # aux = dr.zeros(Array3f, 1)  # Inicializa un vector auxiliar si no se proporciona uno

        aux[0] = self.point_wall_i[0] + (((float(y) / self.height) * self.wallViewWidth + self.pxHalfWidth) * self.wallDirection[0])
        aux[1] = self.point_wall_i[1] + (((float(y) / self.height) * self.wallViewWidth + self.pxHalfWidth) * self.wallDirection[1])
        aux[2] = self.point_wall_i[2] + (((float(y) / self.height) * self.wallViewWidth + self.pxHalfWidth) * self.wallDirection[2])
        return aux
        # ratio = float(y) / self.height
        # offset = ratio * self.wallViewWidth + self.pxHalfWidth
        # return offset * self.wallDirection + self.point_wall_i
        
    # Devuelve los puntos correspondientes a las coordenadas y
    def getPointsForCoord(self, y: np.ndarray, offsetCalc: np.ndarray = None) -> np.ndarray:
        aux = np.zeros((len(y), 3), dtype=float)

        # Si no estaba optimizado el calculo del offset
        if offsetCalc is None:
            # Inicializar offsetCalc si es None
            offsetCalc = np.zeros((len(y), 3), dtype=float)
            # Dividir todas las componentes de y por la altura de la imagen
            ratio = y / self.height
            offset = ratio * self.wallViewWidth + self.pxHalfWidth
            
            offsetCalc[:,0] = offset * self.wallDirection[0]
            offsetCalc[:,1] = offset * self.wallDirection[1]
            offsetCalc[:,2] = offset * self.wallDirection[2]

        # Calcular los puntos de la pared
        aux[:, 0] = offsetCalc[:, 0] + self.point_wall_i[0]
        aux[:, 1] = offsetCalc[:, 1] + self.point_wall_i[1]
        aux[:, 2] = offsetCalc[:, 2] + self.point_wall_i[2]

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
