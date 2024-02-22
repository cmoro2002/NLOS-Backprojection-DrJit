class BoxBounds:
    def __init__(self, xi, yi, zi, scale, resolution):
        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.sx = scale
        self.sy = scale
        self.sz = scale
        self.resolution = resolution

    # def __init__(self, xi, yi, zi, sx, sy, sz, resolution):
    #     self.xi = xi
    #     self.yi = yi
    #     self.zi = zi
    #     self.sx = sx
    #     self.sy = sy
    #     self.sz = sz
    #     self.resolution = resolution
