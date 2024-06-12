import numpy as np

import drjit as dr

from drjit.llvm import Array3f, Float
# Clase que contiene los parámetros de la voxelización transitoria

class TransientVoxelizationParams:
    def __init__(self):
        # Ortho matrix size
        self.orthoMatrix = None
        self.voxelSize = 0
        self.ORTHO_OFFSETX = -0.1
        self.ORTHO_SIZEX = 0.6
        self.ORTHO_OFFSETY = -0.35
        self.ORTHO_SIZEY = 0.6
        self.ORTHO_OFFSETZ = 0.9
        self.ORTHO_SIZEZ = 0.6

        # Geometry config
        self.streakYratio = 1
        self.fov = np.radians(90)
        self.camera = np.array([-0.2, 0, 1]) 
        self.lookTo = np.array([-0.2, 0, 0]) 
        self.laserOrigin = np.array([-0.2, 0, 1]) 
        self.wallNormal = np.array([0, 0, 1]) 
        self.wallDir = np.array([1, 0, 0])
        self.t_delta = 0.001
        self.t0 = 0
        self.UNWARP_LASER = False
        self.UNWARP_CAMERA = False

        self.lasers = Float(0.2, 0, 0)

        # Input
        self.inputFolder = "../../2016_LookingAroundCorners/bunny_final_multilaser_2b_highres"
        self.OPTIM = False
        self.resultsRoute = "resultado"
        self.manual = False
        self.dataset = "dataset/Z.hdf5"
        self.verbose = False
        self.scaleDownTo = None

    def setOrthoSize(self, size):
        self.ORTHO_SIZEX = self.ORTHO_SIZEY = self.ORTHO_SIZEZ = size

    def getMaxOrthoSize(self):
        return max(self.ORTHO_SIZEX, self.ORTHO_SIZEY, self.ORTHO_SIZEZ)

    def validate(self):
        # Ensure intensity multiplier is in range
        self.MAX_INTENSITY_MULTIPLIER = max(1, min(self.MAX_INTENSITY_MULTIPLIER, 255))
        self.wallNormal /= dr.norm(self.wallNormal)
        self.wallDir /= dr.norm(self.wallDir)
        self.ELLIPSOIDS_PER_PIXEL = max(1, self.ELLIPSOIDS_PER_PIXEL)
