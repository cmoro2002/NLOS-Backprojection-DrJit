from TransientImage import TransientImage
from TransientVoxelizationParams import TransientVoxelizationParams
from HDRDecoder import decodeHDRFile
from StreakLaser import StreakLaser

import drjit as dr

# Imports de drjit
# from drjit.cuda import Float, UInt32
from drjit.llvm import Float, Int, Array3f

import argparse
import numpy as np

def parseArgsIntoParams(params):

     # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Descripción de tu programa")

    # Argumentos de línea de comandos
    parser.add_argument("-folder", dest="inputFolder", type=str, help="Nombre de la carpeta")
    parser.add_argument("-fov", type=float)
    parser.add_argument("-voxelRes", type=int)
    parser.add_argument("-ortho", nargs=4, type=float)
    parser.add_argument("-cam", nargs=3, type=float)
    parser.add_argument("-lookTo", nargs=3, type=float)
    parser.add_argument("-laserOrigin", nargs=3, type=float)
    parser.add_argument("-t_delta", type=float)
    parser.add_argument("-lasers", nargs=3, type=float, help="Componentes de los láseres")
    parser.add_argument("-laser_origin", nargs=3, type=float, help="Componentes del origen del láser")
    parsed_args = parser.parse_args()

    # Asignar los argumentos a los parámetros
    if parsed_args.inputFolder is not None:
        params.inputFolder = parsed_args.inputFolder
    if parsed_args.fov is not None:
        params.fov = np.radians(parsed_args.fov)
    if parsed_args.voxelRes is not None:
        params.VOXEL_RESOLUTION = parsed_args.voxelRes
    if parsed_args.ortho is not None:
        params.ORTHO_OFFSETX = parsed_args.ortho[0]
        params.ORTHO_OFFSETY = parsed_args.ortho[1]
        params.ORTHO_OFFSETZ = parsed_args.ortho[2]
        size = parsed_args.ortho[3]
        params.ORTHO_SIZEX = size
        params.ORTHO_SIZEY = size
        params.ORTHO_SIZEZ = size
    if parsed_args.cam is not None:
        params.camera = Float(parsed_args.cam)
    if parsed_args.lookTo is not None:
        params.lookTo = Float(parsed_args.lookTo)
    if parsed_args.laserOrigin is not None:
        params.laserOrigin = Float(parsed_args.laserOrigin)
    if parsed_args.t_delta is not None:
        params.t_delta = parsed_args.t_delta
    if parsed_args.lasers is not None:
        params.lasers = Float(parsed_args.lasers)
    if parsed_args.laser_origin is not None:
        params.laserOrigin = Float(parsed_args.laser_origin)


def setParamsForCamera(params: TransientVoxelizationParams, transient_image: TransientImage, streakLaser: StreakLaser):
    wallDir = Float(1, 0, 0)
    wallNormal = Float(0, 0, 1)
    dwall = dr.sqrt(np.linalg.norm(np.subtract(params.camera, params.lookTo))**2)
    semiWidth = np.tan(params.fov / 2) * dwall
    pxHalfHeight = semiWidth * params.streakYratio / transient_image.height
    streakAbsY = Float(semiWidth - pxHalfHeight - (streakLaser.streak * params.streakYratio / transient_image.height) * semiWidth * 2)

    params.wallDirection = wallDir
    params.wallNormal = wallNormal
    # wall_up = np.cross(transient_image.wallNormal, transient_image.wallDirection)
    wall_up = dr.cross(transient_image.wallNormal, transient_image.wallDirection)

    pointWallI = Array3f(params.lookTo)
    pointWallI += wall_up * streakAbsY
    wall_up = wall_up * -semiWidth
    pointWallI += wall_up

    transient_image.wallViewWidth = semiWidth * 2
    transient_image.pxHalfWidth = transient_image.wallViewWidth / (transient_image.height * 2)

    dwallstreaksq = dwall**2 + streakAbsY**2

    if not params.UNWARP_CAMERA:
        for i in range(len(transient_image.wallCameraDilation)):
            x = (i / len(transient_image.wallCameraDilation)) * transient_image.wallViewWidth - semiWidth + transient_image.pxHalfWidth
            transient_image.wallCameraDilation[i] = np.sqrt(x**2 + dwallstreaksq)

    transient_image.laser = params.lasers

def initTransientImage(params: TransientVoxelizationParams, file_name: str):

    # TODO: Definir bien el streaklaser
    streakLaser = StreakLaser(0, 0)

    transient_image = decodeHDRFile(file_name)

    setParamsForCamera(params, transient_image, streakLaser)

    laserDist = np.sqrt(np.linalg.norm(transient_image.getLaser() - params.laserOrigin)**2)
    transient_image.setLaserHitTime(0 if params.UNWARP_CAMERA else laserDist)


    return transient_image




