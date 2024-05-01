"""
Nombre del archivo: Backprojection.py
Descripción: Implementación de la función de backprojection 
Autor: César Moro Latorre
Fecha de creación: 04/02/2024
Última modificación: 29/04/2024

Ejecución: python3 Backprojection.py -folder <nombre_carpeta> -voxel_resolution <resolución_voxel> -max_ortho_size <tamaño_ortho>
    python3 Backprojection.py -folder letter_ht_90 -voxelRes 256 -lasers 0.2 0 0 -laser_origin 0.2 0 1 -cam 0.2 0 1 -lookTo 0.2 0 0 -fov 90 -t_delta 0.005 -ortho -0.1 -0.35 0.9 0.7 
"""


# Imports de librerias
import time
import argparse
import os
import numpy as np
import drjit as dr

# Imports de drjit
from drjit.llvm import Float, Int, Array3f, UInt32, Loop, UInt
import threading

from typing import List
import matplotlib.pyplot as plt

# Imports de mis clases
from TransientImage import TransientImage
from TransientVoxelization import initTransientImage, parseArgsIntoParams
from BoxBounds import BoxBounds
from TransientVoxelizationParams import TransientVoxelizationParams
from FilterResults import apply_filters

def sumTransientIntensitiesFor(fx: float, fy: float, fz: float, transient_images: List[TransientImage]) -> float:
    voxel = Array3f([fx, fy, fz])
    intensities = 0

    for transient_image in transient_images:
        for h in range(transient_image.height):

            # Obtener el punto de la pared
            wall_point = transient_image.getPointForCoord(h)

            # Obtener la posicion del laser
            laser = transient_image.getLaser()
            
            # Calcular la distancia r2 (pared-voxel)
            r2 = np.sqrt(np.sum(np.square(laser - voxel)))
            
            # Calcular la distancia r3 (voxel-puntos de la pared)
            r3 = np.sqrt(np.sum(np.square(voxel - wall_point)))
            
            time =  r2 + r3
                
            # Sumar la intensidad correspondiente al tiempo
            intensities += transient_image.getIntensityForTime(h, time)

    return intensities

def sumTransientIntensitiesForOptim1(fx: float, fy: float, fz: float, transient_images: List[TransientImage]) -> float:
    voxel = Array3f([fx, fy, fz])
    altura = transient_images[0].height
    # intensities = dr.zeros(Float,altura)
    intensities = np.zeros(altura)

    alturas = np.arange(0, transient_images[0].height)
    laser = transient_images[0].getLaser()

    # Calcular la distancia entre el láser y el voxel
    # laser_voxel_distance = np.sqrt(np.sum(np.square(laser - voxel)))
    laser_voxel_distance = np.linalg.norm(laser - voxel)

    for transient_image in transient_images:
        # Obtener el punto de la pared (no estás utilizando esto en el cálculo de intensidades)
        wall_points = transient_image.getPointsForCoord(alturas)
        # Calcular el tiempo

        # Calcular la distancia entre el voxel y los puntos de la pared
        # voxel_wall_distance = np.sqrt(np.sum(np.square(voxel - wall_points), axis=1))
        voxel_wall_distance = np.linalg.norm(voxel - wall_points, axis=1)

        # Sumar las dos distancias para obtener el tiempo
        times = laser_voxel_distance + voxel_wall_distance

        # Sumar la intensidad correspondiente al tiempo
        intensities += transient_image.getIntensitiesForTime(alturas, times)

    return np.sum(intensities)

def sumTransientIntensitiesForOptim(fx: float, fy: float, fz: float, transient_images: List[TransientImage]) -> float:
    voxel = np.array([fx, fy, fz])
    altura = transient_images[0].height

    # Obtener las alturas y la distancia láser-voxel
    alturas = np.arange(0, altura)

    # r2 (128 distancias)
    r2 = np.sqrt(np.sum(np.square(voxel - transient_images[0].laser)))

    # Calcular las distancias voxel-pared para todas las imágenes y alturas
    r3 = []
    for transient_image in transient_images:
        r3.append(np.sqrt(np.sum((voxel - transient_image.wallPoints)**2, axis=1)))
    
    # r3 (128 imagenes, 128 distancias cada una)

    # Calcular los tiempos y sumar las intensidades
    times = r2 + r3
    intensities = []
    for transient_image, tiempos in zip(transient_images, times):
        intensities.append(transient_image.getIntensitiesForTime(alturas, tiempos))

    return np.sum(intensities)

import numpy as np
from typing import List

def sumTransientIntensitiesForOptim2(fx: float, fy: float, fz: float, transient_images: List[TransientImage]) -> float:
    voxels = np.array([fx, fy, fz])
    altura = transient_images[0].height

    # Obtener las alturas y la distancia láser-voxel
    alturas = np.arange(0, altura)

    # r2 (128 distancias)
    r2 = np.sqrt(np.sum(np.square(voxels - transient_images[0].laser), axis=1))

    # Calcular las distancias voxel-pared para todas las imágenes y alturas
    r3 = np.array([np.sqrt(np.sum((voxels - ti.wallPoints)**2, axis=1)) for ti in transient_images])

    # r3 (128 imagenes, 128 distancias cada una)

    # Calcular los tiempos y sumar las intensidades
    times = r2[:, np.newaxis] + r3
    intensities = np.sum([ti.getIntensitiesForTime(alturas, tiempos) for ti, tiempos in zip(transient_images, times)], axis=0)

    return np.sum(intensities)


def setWallPoints(transient_images: List[TransientImage]):
    alturas = np.arange(0, transient_images[0].height)
    aux = np.zeros((len(alturas), 3), dtype=float)

    # Dividir todas las componentes de y por la altura de la imagen
    ratio = alturas / transient_images[0].height
    offset = ratio * transient_images[0].wallViewWidth + transient_images[0].pxHalfWidth
    
    aux[:, 0] = offset * transient_images[0].wallDirection[0] 
    aux[:, 1] = offset * transient_images[0].wallDirection[1]
    aux[:, 2] = offset * transient_images[0].wallDirection[2]

    for transient_image in transient_images:

        wall_points = np.zeros((len(alturas), 3), dtype=float)

        wall_points[:, 0] = aux[:, 0] + transient_image.point_wall_i[0]
        wall_points[:, 1] = aux[:, 1] + transient_image.point_wall_i[1]
        wall_points[:, 2] = aux[:, 2] + transient_image.point_wall_i[2]

        transient_image.setWallPoints(wall_points)

def backprojection(params: TransientVoxelizationParams):

    # Crear una instancia de TransientImage
    transient_images = initTransientImages(params)

    folder_name = params.inputFolder
    if (params.OPTIM):
        print(f"Empezando el proceso de backprojection optimizado para de la carpeta {folder_name}")
    else:
        print(f"Empezando el proceso de backprojection para de la carpeta {folder_name}")

    bounds = BoxBounds(params.ORTHO_OFFSETX, params.ORTHO_OFFSETY, params.ORTHO_OFFSETZ, params.getMaxOrthoSize(), params.VOXEL_RESOLUTION)

    resolution = bounds.resolution

    results = np.zeros((resolution, resolution, resolution))

    start_time = time.time()

    # Calcular los wallpoints de cada imagen 
    if params.OPTIM:
        setWallPoints(transient_images)
        print(f"WallPoints calculados")
    
    for z in range(resolution):
        start_time_z = time.time()  # Registrar el tiempo de inicio de la iteración
        for y in range(resolution):
            for x in range(resolution):

                fx = bounds.xi + ((x + 0.5) / resolution) * bounds.sx
                fy = bounds.yi + ((y + 0.5) / resolution) * bounds.sy
                fz = bounds.zi + ((z + 0.5) / resolution) * bounds.sz

                if params.OPTIM:
                    # Almacenar la suma de los resultados
                    results[y, x, z] = sumTransientIntensitiesForOptim(fx, fy, fz, transient_images)
                else:
                    # Almacenar la suma de los resultados
                    results[y, x, z] += sumTransientIntensitiesFor(fx, fy, fz, transient_images)

        end_time_z = time.time()  # Registrar el tiempo de finalización de la iteración
        elapsed_time = end_time_z - start_time_z 
        print(f"Iteración z={z} tarda {elapsed_time} segundos")

    
    # results = vectorize_transient_intensities(bounds, resolution, transient_images)


    # Crear matrices de coordenadas voxel    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"El proceso de backprojection ha tardado {elapsed_time} segundos")

    # Guardar los resultados en un fichero:
    print(f"Guardando resultados en results/results")
    FilterResults = apply_filters(resolution, results)

    # Visualizar los resultados
    plt.imshow(FilterResults.max_result, cmap='hot', interpolation='nearest')

    # flattened_results = results[:,:,z]
    # plt.imshow(flattened_results, cmap='hot', interpolation='nearest')
    plt.colorbar()
    # Guardar la imagen en results/params.resultsRoute
    plt.savefig('results/' + params.resultsRoute + '.png') 
    plt.show()
    
    print(f"Proceso de backprojection finalizado")


# Recorrer la lista de imagenes de la carpeta y crear una instancia de TransientImage por cada imagen
def initTransientImages(params: TransientVoxelizationParams):

    # Obtener la lista de archivos en la carpeta
    files = os.listdir(params.inputFolder)

    # Ordenar la lista de archivos en orden alfabético
    files.sort()

    count = 0
    transient_images = []

    print(f"Se han encontrado {len(files)} archivos en la carpeta {params.inputFolder}")
    
    # Recorrer los archivos y crear una instancia de TransientImage por cada imagen
    for file in files:
        # Comprobar si el archivo es una imagen (puedes ajustar esta comprobación según tus necesidades)
        if file.endswith(('.hdr')):
            file_route = params.inputFolder + "/" + file
            transient_image = initTransientImage(params, file_route, count)
            
            # Agregar la instancia de TransientImage a la lista
            transient_images.append(transient_image)

            count += 1
        else :
            print(f"El archivo {file} no es una imagen .hdr")

    print(f"Se han leido un total de {count} de imagenes")
    return transient_images

def main():

    # Configuración de la variable de entorno para seleccionar la GPU a utilizar
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Parsear los argumentos de línea de comandos
    params = TransientVoxelizationParams()
    parseArgsIntoParams(params)

    print(f"Argumentos de línea de comandos leidos")
    backprojection(params)


if __name__ == "__main__":
    # Ejecuta el código principal
    main()