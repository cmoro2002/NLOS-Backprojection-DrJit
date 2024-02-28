from TransientImage import TransientImage
from TransientVoxelizationParams import TransientVoxelizationParams
from HDRDecoder import decodeHDRFile
from StreakLaser import StreakLaser

# Imports de drjit
from drjit.cuda import Float, Int

import argparse
import numpy as np

def parseArgsIntoParams(params, args):

     # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Descripción de tu programa")

    # Argumentos de línea de comandos
    parser.add_argument("-folder", dest="folder_name", type=str, help="Nombre de la carpeta")
    parser.add_argument("-fov", type=float)
    parser.add_argument("-voxelRes", type=int)
    parser.add_argument("-ortho", nargs=4, type=float)
    parser.add_argument("-cam", nargs=3, type=float)
    parser.add_argument("-lookTo", nargs=3, type=float)
    parser.add_argument("-laserOrigin", nargs=3, type=float)
    parser.add_argument("-t_delta", type=float)
    parser.add_argument("-lasers", nargs='+', type=float, help="Componentes de los láseres")
    parsed_args = parser.parse_args(args)

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
        params.camera = np.array(parsed_args.cam, dtype=np.float32)
    if parsed_args.lookTo is not None:
        params.lookTo = np.array(parsed_args.lookTo, dtype=np.float32)
    if parsed_args.laserOrigin is not None:
        params.laserOrigin = np.array(parsed_args.laserOrigin, dtype=np.float32)
    if parsed_args.t_delta is not None:
        params.t_delta = parsed_args.t_delta
    if parsed_args.lasers is not None:
        params.lasers = np.array(parsed_args.lasers, dtype=np.float32)


def setParamsForCamera(params: TransientVoxelizationParams, transient_image: TransientImage, streakLaser: StreakLaser):
    wallDir = Float([1, 0, 0])
    wallNormal = Float([0, 0, 1])
    dwall = np.sqrt(np.linalg.norm(np.subtract(params.camera, params.lookTo))**2)
    semiWidth = np.tan(params.fov / 2) * dwall
    pxHalfHeight = semiWidth * params.streakYratio / params.height
    streakAbsY = semiWidth - pxHalfHeight - (streakLaser.streak * params.streakYratio / params.height) * semiWidth * 2

    params.wallDirection = wallDir
    params.wallNormal = wallNormal
    wall_up = np.cross(params.wallNormal, params.wallDirection)

    pointWallI = params.lookTo.copy()
    pointWallI += wall_up * streakAbsY
    wall_up *= -semiWidth
    pointWallI += wall_up

    params.wallViewWidth = semiWidth * 2
    params.pxHalfWidth = params.wallViewWidth / (params.height * 2)

    dwallstreaksq = dwall**2 + streakAbsY**2

    if not params.UNWARP_CAMERA:
        for i in range(len(params.wallCameraDilation)):
            x = (i / len(params.wallCameraDilation)) * params.wallViewWidth - semiWidth + params.pxHalfWidth
            params.wallCameraDilation[i] = np.sqrt(x**2 + dwallstreaksq)

def initTransientImage(params: TransientVoxelizationParams, file_name: str):



    # TODO: Definir StreakLaser a partir del file
    streakLaser = StreakLaser(0, 0)

    transient_image = decodeHDRFile(file_name)

    # TODO: Definir parametros de la camara
    setParamsForCamera(params, transient_image, streakLaser)

    # TODO: Definir distancia al laser  y su hit time
    laserDist = np.sqrt(np.linalg.norm(transient_image.getLaser() - params.laserOrigin)**2)
    transient_image.setLaserHitTime(0 if params.UNWARP_CAMERA else laserDist)


    return transient_image




