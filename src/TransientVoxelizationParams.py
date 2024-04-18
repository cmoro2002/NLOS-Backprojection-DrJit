import numpy as np

import drjit as dr

from drjit.llvm import Array3f, Float
# Clase que contiene los parámetros de la voxelización transitoria

class TransientVoxelizationParams:
    def __init__(self):
        # General params
        self.VERBOSE = False
        self.VOXEL_RESOLUTION = 128
        self.DELAYED_RENDER_BATCH_SIZE = 160000
        self.SPHERE_MAX_RECURSIONS = 7
        self.MEMORY_SAVING_MODE = False
        self.MAX_INTENSITY_MULTIPLIER = 255
        self.ERROR_THRESHOLD_WEIGHT = 1
        self.ELLIPSOIDS_PER_PIXEL = 1
        self.STOCHASTIC = False
        self.ELLIPSOID_PER_PIXEL_THRESHOLD_WEIGHT = 1
        self.USE_HALF_ELLIPSOIDS = True
        self.ALLOW_OVERFLOW_PROTECTION = True
        self.CLAMP_INTENSITY_GREATER_THAN = -1
        self.NORMALIZE_TO_UNIT_INTERVAL = True

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
        self.READ_HDR_FILES = True
        self.lasersFile = None
        self.readLasersAsText = False
        self.wallFile = None
        self.readWallAsText = False

        # Output
        self.DEFAULT_SAVE_AS_HDR = False
        self.saveFolder = None
        self.saveImage = False
        self.filename2d = None
        self.save2DRaw = False
        self.filename2draw = None
        self.printGrayscale = False
        self.save3DDump = False
        self.filename3d = None
        self.save3DRaw = False
        self.filename3draw = None
        self.backprojectCpu = True
        self.executionInfoFile = None
        self.PRINT_TRANSIENT_IMAGES = False
        self.OVERRIDE_TRANSIENT_WALL_POINTS = None

        # Custom, not input-assignable params
        self.AUTO_CLEAN = True
        self.FORCE_2D_BACKPROJECT = False
        self.ENABLE_HARDWARE_CONSERVATIVE_RASTER = False
        self.CLEAR_STORAGE_ON_INIT = True
        self.CUSTOM_TRANSIENT_IMAGES = None
        self.AUTO_MANAGE_DISPLAY = False

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
