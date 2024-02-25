import numpy as np

class TransientImage:
    def __init__(self, width: int, height: int, channels: int, time_per_coord: float, intensity_multiplier_unit: float, data: np.ndarray, max_value: float, min_value: float):
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
    def get_intensity_for_time(self, y: int, time: float) -> float:
        x = int((time + self.laser_hit_time + self.wall_camera_dilation[y]) / self.time_per_coord)
        if x >= self.width or x < 0:
            return 0
        return self.data[x][y][0]

    def __str__(self):
        return f"TransientImage: width={self.width}, height={self.height}, channels={self.channels}, time_per_coord={self.time_per_coord}, intensity_multiplier_unit={self.intensity_multiplier_unit}, max_value={self.max_value}, min_value={self.min_value}"
