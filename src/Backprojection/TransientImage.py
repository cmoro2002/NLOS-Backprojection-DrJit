from typing import List
from mathutils import Vector

class TransientImage:
    def __init__(self, width: int, height: int, channels: int, time_per_coord: float, intensity_multiplier_unit: float, data: List[List[List[float]]], max_value: float, min_value: float):
        self.width = width
        self.height = height
        self.channels = channels
        self.time_per_coord = time_per_coord
        self.intensity_multiplier_unit = intensity_multiplier_unit
        self.data = data if data is not None else [[[0.0] * channels for _ in range(height)] for _ in range(width)]
        self.max_value = max_value
        self.min_value = min_value
        self.laser_hit_time = 0
        self.wall_camera_dilation = [0.0] * height
        self.point_wall_i = Vector((0, 0, 0))
        self.wall_direction = Vector((0, 0, 0))
        self.wall_normal = Vector((0, 0, 0))
        self.laser = Vector((0, 0, 0))
        self.wall_view_width = height
        self.px_half_width = 0.5
        
    def __str__(self):
        return f"TransientImage: width={self.width}, height={self.height}, channels={self.channels}, time_per_coord={self.time_per_coord}, intensity_multiplier_unit={self.intensity_multiplier_unit}, max_value={self.max_value}, min_value={self.min_value}"
