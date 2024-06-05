import drjit as dr
from drjit.llvm import Float

class BackProjectionParams:
    def __init__( self, laserWallPos, t0, t_delta, data, r1, r4, wallPoints, hiddenVolumePosition, hiddenVolumeSize, results) :
        self.laserWallPos = laserWallPos # Posicion del laser
        self.t0 = t0 # Tiempo inicial
        self.t_delta = t_delta # Tiempo entre frames
        self.r1 = r1 # Distancia entre el origen del laser y el punto al que apunta en la pared
        self.r4 = r4 # Distancias desde la camara y los puntos de la pared
        self.wallPoints = wallPoints # Puntos de la pared
        self.hiddenVolumePosition = hiddenVolumePosition
        self.hiddenVolumeSize = hiddenVolumeSize
        self.width = data.shape[0]
        self.height = data.shape[1]
        self.depth = data.shape[2]
        self.data = results # Datos del dataset en formato vector, no matriz
    
    def to_string(self):
        return (
            f"BackprojectionParams:\n"
            f"  t0: {self.t0}\n"
            f"  t_delta: {self.t_delta}\n"
            f"  data.shape: {self.data}\n"
            f"  r1: {self.r1.Shape}\n"
            f"  r4: {self.r4.Shape}\n"
            f"  wallPoints: {self.wallPoints.Shape}\n"
            f"  hiddenVolumePosition: {self.hiddenVolumePosition}\n"
            f"  hiddenVolumeSize: {self.hiddenVolumeSize}\n"
            f"  width: {self.width}\n"
            f"  height: {self.height}\n"
            f"  depth: {self.depth}\n"
        )