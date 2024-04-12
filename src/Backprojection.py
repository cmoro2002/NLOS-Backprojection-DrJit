"""
Nombre del archivo: Backprojection.py
Descripción: Implementación de la función de backprojection 
Autor: César Moro Latorre
Fecha de creación: 04/02/2024
Última modificación: 29/04/2024

Ejecución: python3 Backprojection.py -folder <nombre_carpeta> -voxel_resolution <resolución_voxel> -max_ortho_size <tamaño_ortho>
    $ python3 Backprojection.py -folder letter_ht_90 -voxelRes 256 -lasers 0.2 0 0 -laser_origin 0.2 0 1 -cam 0.2 0 1 -lookTo 0.2 0 0 -fov 90 -t_delta 0.005 -ortho -0.1 -0.35 0.9 0.7 
"""


# Imports de librerias
import time
import argparse
import os
import numpy as np
import drjit as dr

# Imports de drjit
from drjit.llvm import Float, Int, Array3f, UInt32, Loop, UInt

from typing import List
import matplotlib.pyplot as plt

# Imports de mis clases
from TransientImage import TransientImage
from TransientVoxelization import initTransientImage, parseArgsIntoParams
from BoxBounds import BoxBounds
from TransientVoxelizationParams import TransientVoxelizationParams

# def sumTransientIntensitiesFor(fx: Float, fy: Float, fz: Float, transient_images: List[TransientImage]) -> Float:
#     voxel = Array3f([fx, fy, fz])
#     altura = transient_images[0].height
#     intensities = dr.zeros(Float,altura)

#     for transient_image in transient_images:

#         # Construir un vector con todos los valores desde 1 hasta la altura de la imagen
#         alturas = np.arange(1, transient_image.height)
#         result = transient_image.getPointsForCoord(alturas)

#         for h in range(transient_image.height):

#             # Obtener el punto de la pared (no estás utilizando esto en el cálculo de intensidades)
#             wall_point = transient_image.getPointForCoord(h)

#             # Calcular el tiempo
#             laser = transient_image.getLaser()
            
#             # Calcular la distancia entre el láser y el voxel
#             laser_voxel_distance = dr.sqrt(np.sum(np.square(laser - voxel)))
            
#             # Calcular la distancia entre el voxel y los puntos de la pared
#             voxel_wall_distance = dr.sqrt(np.sum(np.square(voxel - wall_point)))

#             print(f"Distancia entre láser y voxel: {laser_voxel_distance}")
#             print(f"Distancia entre voxel y punto de la pared: {voxel_wall_distance}")
            
#             time =  laser_voxel_distance + voxel_wall_distance
                
#             # Sumar la intensidad correspondiente al tiempo
#             valor = transient_image.getIntensityForTime(h, time)

#             intensities[h] = valor

#     return dr.sum(intensities)

def sumTransientIntensitiesFor(fx: Float, fy: Float, fz: Float, transient_images: List[TransientImage]) -> Float:
    voxel = Array3f([fx, fy, fz])
    altura = transient_images[0].height
    intensities = dr.zeros(Float,altura)

    for transient_image in transient_images:
        alturas = np.arange(1, transient_image.height)
        # Obtener el punto de la pared (no estás utilizando esto en el cálculo de intensidades)
        wall_points = transient_image.getPointsForCoord(alturas)
       
        # Calcular el tiempo
        laser = transient_image.getLaser()  # Suponiendo que este es un vector

        # Calcular la distancia entre el láser y el voxel
        laser_voxel_distance = np.sqrt(np.sum(np.square(laser - voxel)))

        # Calcular la distancia entre el voxel y los puntos de la pared
        voxel_wall_distance = np.sqrt(np.sum(np.square(voxel - wall_points), axis=1))

        # Sumar las dos distancias para obtener el tiempo
        times = laser_voxel_distance + voxel_wall_distance

        # Sumar la intensidad correspondiente al tiempo
        intensities = transient_image.getIntensitiesForTime(alturas, times)

    return dr.sum(intensities)

def backprojection(params: TransientVoxelizationParams):

    # Crear una instancia de TransientImage
    transient_images = initTransientImages(params)

    folder_name = params.inputFolder
    print(f"Empezando el proceso de backprojection para de la carpeta {folder_name}")

    bounds = BoxBounds(params.ORTHO_OFFSETX, params.ORTHO_OFFSETY, params.ORTHO_OFFSETZ, params.getMaxOrthoSize(), params.VOXEL_RESOLUTION)

    resolution = bounds.resolution

    print(f"Resolución: {resolution}")
    results = np.zeros((resolution, resolution, resolution))

    # for z in range(resolution):
    z = 0
    for y in range(resolution):
        start_time_y = time.time()  # Registrar el tiempo de inicio de la iteración
        for x in range(resolution):

            fx = bounds.xi + ((x + 0.5) / resolution) * bounds.sx
            fy = bounds.yi + ((y + 0.5) / resolution) * bounds.sy
            fz = bounds.zi + ((z + 0.5) / resolution) * bounds.sz

            # Almacenar la suma de los resultados
            results[x, y, z] += sumTransientIntensitiesFor(fx, fy, fz, transient_images)

        end_time_y = time.time()  # Registrar el tiempo de finalización de la iteración
        elapsed_time = end_time_y - start_time_y 
        print(f"Iteración (y={y}, z={z}) tarda {elapsed_time} segundos")
    print(f"Imagen {z} procesada")

    # Guardar los resultados en un fichero:
    print(f"Guardando resultados en {folder_name}_results")
    np.save(f"{folder_name}_results", results)

    flattened_results = results[:,:,0]
    plt.imshow(flattened_results, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()
    
    plt.savefig('resultado_visualizacion.png')  # Puedes especificar el nombre del archivo y la extensión que desees

    
    print(f"Proceso de backprojection finalizado")


# Recorrer la lista de imagenes de la carpeta y crear una instancia de TransientImage por cada imagen
def initTransientImages(params: TransientVoxelizationParams):

    # Obtener la lista de archivos en la carpeta
    files = os.listdir(params.inputFolder)

    # Ordenar la lista de archivos en orden alfabético
    files.sort()

    count = 0
    transient_images = []
    
    # Recorrer los archivos y crear una instancia de TransientImage por cada imagen
    for file in files:
        # Comprobar si el archivo es una imagen (puedes ajustar esta comprobación según tus necesidades)
        if file.endswith(('.hdr')):
            file_route = params.inputFolder + "/" + file
            print(f"Reading file: {file_route}")
            transient_image = initTransientImage(params, file_route)
            
            # Agregar la instancia de TransientImage a la lista
            transient_images.append(transient_image)

            count += 1

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
    # Registra el tiempo de inicio
    start_time = time.time()

    # Ejecuta el código principal
    main()

    # Registra el tiempo de finalización
    end_time = time.time()

    # Calcula el tiempo total de ejecución
    execution_time = end_time - start_time

    # Muestra el tiempo total de ejecución
    print(f"Tiempo de ejecución: {execution_time} segundos")