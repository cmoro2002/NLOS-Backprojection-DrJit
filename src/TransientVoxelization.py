from TransientImage import TransientImage
from TransientVoxelizationParams import TransientVoxelizationParams
from HDRDecoder import decodeHDRFile
from StreakLaser import StreakLaser

import drjit as dr
import copy

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
    parser.add_argument("-Optim", help="Optimizar el código")
    parser.add_argument("-resultsRoute", type=str, help="Nombre del archivo donde guardar los resultados")
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
        params.camera = np.array(parsed_args.cam)
    if parsed_args.lookTo is not None:
        params.lookTo = np.array(parsed_args.lookTo)
    if parsed_args.laserOrigin is not None:
        params.laserOrigin = np.array(parsed_args.laserOrigin)
    if parsed_args.t_delta is not None:
        params.t_delta = parsed_args.t_delta
    if parsed_args.lasers is not None:
        params.lasers = np.array(parsed_args.lasers)
    if parsed_args.laser_origin is not None:
        params.laserOrigin = np.array(parsed_args.laser_origin)
    if parsed_args.Optim is not None:
        params.OPTIM = True
    if parsed_args.resultsRoute is not None:
        params.resultsRoute = parsed_args.resultsRoute



def setParamsForCamera(params: TransientVoxelizationParams, transient_image: TransientImage, streak: int):
    wallDir = np.array([1, 0, 0])
    wallNormal = np.array([0, 0, 1])
    dwall = np.sqrt(np.linalg.norm(np.subtract(params.camera, params.lookTo))**2)
    semiWidth = np.tan(params.fov / 2) * dwall
    pxHalfHeight = semiWidth * params.streakYratio / transient_image.height


    streakAbsY = float(semiWidth - pxHalfHeight - (streak * params.streakYratio / transient_image.height) * semiWidth * 2)

    transient_image.wallDirection = copy.deepcopy(wallDir)
    transient_image.wallNormal = copy.deepcopy(wallNormal)
    wall_up = np.cross(transient_image.wallNormal, transient_image.wallDirection)
    # wall_up = dr.cross(transient_image.wallNormal, transient_image.wallDirection)

    transient_image.point_wall_i = copy.deepcopy(params.lookTo)
    transient_image.point_wall_i += wall_up * streakAbsY

    wall_up = transient_image.wallDirection * -semiWidth
    transient_image.point_wall_i += wall_up

    transient_image.wallViewWidth = semiWidth * 2
    transient_image.pxHalfWidth = transient_image.wallViewWidth / (transient_image.height * 2)
    dwallstreaksq = dwall**2 + streakAbsY**2

    if not params.UNWARP_CAMERA:
        for i in range(len(transient_image.wallCameraDilation)):
            x = ((i / len(transient_image.wallCameraDilation)) * transient_image.wallViewWidth) - semiWidth + transient_image.pxHalfWidth
            transient_image.wallCameraDilation[i] = np.sqrt(x**2 + dwallstreaksq) - params.t0
    transient_image.laser = params.lasers



def initTransientImage(params: TransientVoxelizationParams, file_name: str, streak: int):

    # TODO: Definir bien el streaklaser
    # streakLaser = StreakLaser(0, 0)

    transient_image = decodeHDRFile(file_name)

    transient_image.time_per_coord = params.t_delta

    setParamsForCamera(params, transient_image, streak)

    laserDist = np.sqrt(np.linalg.norm(transient_image.getLaser() - params.laserOrigin)**2)

    transient_image.setLaserHitTime(0 if params.UNWARP_CAMERA else laserDist)


    return transient_image




