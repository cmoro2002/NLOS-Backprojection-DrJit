import numpy as np
import drjit as dr

from drjit.cuda import Float, UInt32, Int

class TransientImage:
    def __init__(self, width: Int, height: Int, channels: Int, time_per_coord: Float, intensity_multiplier_unit: Float, data: np.ndarray, max_value: Float, min_value: Float):
        self.width = width
        self.height = height
        self.channels = channels
        self.time_per_coord = time_per_coord
        self.intensity_multiplier_unit = intensity_multiplier_unit
        if data is not None:
            self.data = np.array(data)
        else:
            self.data = np.zeros((width, height, channels))
        self.max_value = max_value
        self.min_value = min_value
        self.laser_hit_time = 0
        self.wall_camera_dilation = np.zeros(height)
        self.point_wall_i = np.array([0, 0, 0])
        self.wall_direction = np.array([0, 0, 0])
        self.wall_normal = np.array([0, 0, 0])
        self.laser = np.array([0, 0, 0])
        self.wall_view_width = height
        self.px_half_width = 0.5

    # Devuelve la intensidad de la imagen en el punto (x, y)
    def getIntensityForTime(self, y: Int, time: Float) -> Float:
        x = Int((time + self.laser_hit_time + self.wall_camera_dilation[y]) / self.time_per_coord)
        if x >= self.width or x < 0:
            return 0
        return self.data[x][y][0]
    
    # Devuelve el punto correspondiente a la coordenada y
    def getPointForCoord(self, y: Int, aux=None):
        if aux is None:
            aux = np.zeros(3)  # Inicializa un vector auxiliar si no se proporciona uno

        aux[0] = self.point_wall_i[0] + (((float(y) / self.height) * self.wall_view_width + self.px_half_width) * self.wall_direction[0])
        aux[1] = self.point_wall_i[1] + (((float(y) / self.height) * self.wall_view_width + self.px_half_width) * self.wall_direction[1])
        aux[2] = self.point_wall_i[2] + (((float(y) / self.height) * self.wall_view_width + self.px_half_width) * self.wall_direction[2])
        
        return aux
    
    # Método para obtener el valor del atributo laser
    def get_laser(self):
        return self.laser
    
    def setLaserHitTime(self, t):
        self.laserHitTime = t

    def getLaser(self):
        return self.laser

    def __str__(self):
        return f"TransientImage: width={self.width}, height={self.height}, channels={self.channels}, time_per_coord={self.time_per_coord}, intensity_multiplier_unit={self.intensity_multiplier_unit}, max_value={self.max_value}, min_value={self.min_value}"
