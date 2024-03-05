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
from drjit.llvm import Float, Int, Array3f

from typing import List
import matplotlib.pyplot as plt

# Imports de mis clases
from TransientImage import TransientImage
from TransientVoxelization import initTransientImage, parseArgsIntoParams
from BoxBounds import BoxBounds
from TransientVoxelizationParams import TransientVoxelizationParams

def sumTransientIntensitiesFor(fx: float, fy: float, fz: float, transient_images: List[TransientImage]):
    voxel = Array3f([fx, fy, fz])
    print(f"Voxel: {voxel}")
    intensities = 0.0

    for transient_image in transient_images:
        for h in range(transient_image.height):

            # Obtener el punto de la pared (no estás utilizando esto en el cálculo de intensidades)
            wall_point = transient_image.getPointForCoord(h)

            # Calcular el tiempo
            time = (np.sqrt(np.sum(np.square(transient_image.getLaser() - voxel))) +
                np.sqrt(np.sum(np.square(voxel - wall_point))))

            # Sumar la intensidad correspondiente al tiempo
            intensities += transient_image.getIntensityForTime(h, time)

    return intensities


def backprojection(params: TransientVoxelizationParams):

    # Crear una instancia de TransientImage
    transient_images = initTransientImages(params)

    folder_name = params.inputFolder
    print(f"Empezando el proceso de backprojection para de la carpeta {folder_name}")

    #TODO: Parametros ORTHO_OFFSET x y z
    bounds = BoxBounds(0, 0, 0, params.getMaxOrthoSize(), params.VOXEL_RESOLUTION)

    num_images = len(transient_images)
    resolution = bounds.resolution

    print(f"Resolución: {resolution}")
    results = np.zeros((resolution, resolution, resolution))

    for z in range(num_images):
        # Para cada x e y de cada imagen 
        for y in range(resolution):
            start_time_y = time.time()  # Registrar el tiempo de inicio de la iteración
            for x in range(resolution):
                start_time_x = time.time()  # Registrar el tiempo de inicio de la iteración

                fx = bounds.xi + ((x + 0.5) / resolution) * bounds.sx
                fy = bounds.yi + ((y + 0.5) / resolution) * bounds.sy
                fz = bounds.zi + ((z + 0.5) / resolution) * bounds.sz

                # Almacenar la suma de los resultados
                results[x, y, z] += sumTransientIntensitiesFor(fx, fy, fz, transient_images)

                end_time_x = time.time()  # Registrar el tiempo de finalización de la iteración
                elapsed_time = end_time_x - start_time_x  # Calcular el tiempo transcurrido en la iteración
                print(f"Iteración (x={x}, y={y}, z={z}) tarda {elapsed_time} segundos")
            end_time_y = time.time()  # Registrar el tiempo de finalización de la iteración
            elapsed_time = end_time_y - start_time_y  #
            print(f"Iteración (y={y}, z={z}) tarda {elapsed_time} segundos")
        print(f"Imagen {z} procesada")

    # Guardar los resultados en un fichero:
    print(f"Guardando resultados en {folder_name}_results")
    np.save(f"{folder_name}_results", results)

    # Mostrar los resultados
    plt.imshow(results[0, :, :])
    plt.show()
    
                 



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