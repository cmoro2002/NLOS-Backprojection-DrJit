# Imports de librerias
import time
import argparse
import os
import numpy as np
import drjit as dr

# Imports de drjit
from drjit.cuda import Float, Int
from typing import List

# Imports de mis clases
from TransientImage import TransientImage
from TransientVoxelization import initTransientImage, parseArgsIntoParams
from BoxBounds import BoxBounds
from TransientVoxelizationParams import TransientVoxelizationParams

def sumTransientIntensitiesFor(fx: Float, fy: Float, fz: Float, transient_images: List[TransientImage]):
    voxel = Float([fx, fy, fz])
    intensities = 0.0

    for transient_image in transient_images:
        for h in range(transient_image.height):
            # Obtener el punto de la pared (no estás utilizando esto en el cálculo de intensidades)
            wall_point = transient_image.getPointForCoord(h)

            # Calcular el tiempo
            time = (dr.sqrt(dr.sum(dr.square(transient_image.getLaser() - voxel))) +
                    dr.sqrt(dr.sum(dr.square(voxel - wall_point))))

            # Sumar la intensidad correspondiente al tiempo
            intensities += transient_image.getIntensityForTime(h, time)

    return intensities


def backprojection(params: TransientVoxelizationParams):

    # Crear una instancia de TransientImage
    transient_images = initTransientImages(params.folder_name)

    folder_name = params.folder_name
    print(f"Empezando el proceso de backprojection para de la carpeta {folder_name}")

    #TODO: Parametros ORTHO_OFFSET x y z
    bounds = BoxBounds(0, 0, 0, params.max_ortho_size, params.resolucion)

    num_images = len(transient_images)
    resolution = bounds.resolution

    results = dr.zeros((resolution, resolution, resolution))

    for z in range(num_images):
        # Para cada x e y de cada imagen 
        for y in range(resolution):
            for x in range(resolution):

                fx = bounds.xi + ((x + 0.5) / resolution) * bounds.sx
                fy = bounds.yi + ((y + 0.5) / resolution) * bounds.sy
                fz = bounds.zi + ((z + 0.5) / resolution) * bounds.sz

                # Almacenar la suma de los resultados
                results[x, y, z] += sumTransientIntensitiesFor(fx, fy, fz, transient_images)
                 



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

            print( f"TransientImage {count}: {transient_image}")
            count += 1

    print(f"Se han leido un total de {count} de imagenes")
    return transient_images

def parse_args():

     # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Descripción de tu programa")
    parser.add_argument("-folder", dest="folder_name", type=str, help="Nombre de la carpeta")
    parser.add_argument("-voxel_resolution", dest="voxel_resolution", type=int, help="Resolución del voxel")
    parser.add_argument("-max_ortho_size", dest="max_ortho_size", type=int, help="Tamaño máximo del ORTHO")

    # Parsear los argumentos de línea de comandos
    args = parser.parse_args()

    # Acceder al valor del argumento -folder
    print(f"reading files from: {folder_name}")

    # Parsear los argumentos de línea de comandos
    params = TransientVoxelizationParams()
    parseArgsIntoParams(params, args)

    return params

def main():

    # Configuración de la variable de entorno para seleccionar la GPU a utilizar
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    params = parse_args()
   
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