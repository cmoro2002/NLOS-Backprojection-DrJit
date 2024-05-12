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
from drjit.llvm import Float, Int, Array3f, UInt32, Loop, UInt, TensorXf
import threading

from typing import List
import matplotlib.pyplot as plt

# Imports de mis clases
from TransientImage import TransientImage
from TransientVoxelization import initTransientImage, parseArgsIntoParams
from BoxBounds import BoxBounds
from TransientVoxelizationParams import TransientVoxelizationParams
from FilterResults import apply_filters

def calcularIndices(x: Int, y: Int, width: int) -> Int:
    return (y * width) + x
    

def sumTransientIntensitiesForOptim(voxeles: Array3f, transient_images: List[TransientImage], wallPoints: Array3f, datos: Float, wallCameraDilations: Float, indicesLectura: Float, resolution: int) -> Float:

    altura = transient_images[0].height

    # Obtener las alturas y la distancia láser-voxel
    alturas = dr.arange(Int,0, altura)
    
    # r2 (128 distancias)
    r2 = dr.norm(voxeles - transient_images[0].laser)

    voxelesR = dr.repeat(voxeles, len(transient_images) * altura)

    # Calcular las distancias voxel-pared para todas las imágenes y alturas
    r3 = dr.norm(voxelesR - wallPoints)

    r2 = dr.repeat(r2, altura * len(transient_images))

    # r3 (128 imagenes, 128 distancias cada una)
    # Calcular los tiempos y sumar las intensidades
    times = r2 + r3

    x = Int((times + transient_images[0].laserHitTime + wallCameraDilations) / transient_images[0].time_per_coord)
    x = dr.clip(x, 0, transient_images[0].width - 1)
    x += indicesLectura

    alturas = dr.tile(alturas, len(transient_images) * resolution * resolution * resolution)

    indices = calcularIndices(x, alturas, transient_images[0].width)

    intensities = dr.gather(Float, datos, indices)

    tam = len(transient_images) * altura

    return dr.block_sum(intensities, tam)

def setWallPoints(transient_images: List[TransientImage]):
    alturas = np.arange(0, transient_images[0].height)
    aux = np.zeros((len(alturas), 3), dtype=float)

    # Dividir todas las componentes de y por la altura de la imagen
    ratio = alturas / transient_images[0].height
    offset = ratio * transient_images[0].wallViewWidth + transient_images[0].pxHalfWidth
    
    aux[:, 0] = offset * transient_images[0].wallDirection[0] 
    aux[:, 1] = offset * transient_images[0].wallDirection[1]
    aux[:, 2] = offset * transient_images[0].wallDirection[2]

    res = dr.zeros(Array3f, len(transient_images) * transient_images[0].height)
    i = 0
    for transient_image in transient_images:

        wall_points = np.zeros((len(alturas), 3), dtype=float)

        wall_points[:, 0] = aux[:, 0] + transient_image.point_wall_i[0]
        wall_points[:, 1] = aux[:, 1] + transient_image.point_wall_i[1]
        wall_points[:, 2] = aux[:, 2] + transient_image.point_wall_i[2]

        DrwallPoints = Array3f(wall_points)

        dr.scatter(res, DrwallPoints, dr.arange(Int, i * transient_images[0].height, (i + 1) * transient_images[0].height))
        i += 1
        wall_pointsDR = Array3f(wall_points)
        transient_image.setWallPoints(wall_pointsDR)

    return res

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

    numVoxeles = resolution * resolution * resolution

    # Calcular los wallpoints de cada imagen 
    wallPoints = setWallPoints(transient_images)
    wallPoints = dr.tile(wallPoints, numVoxeles)
    datos = dr.zeros(Float, transient_images[0].height * transient_images[0].width * len(transient_images))

    wallCameraDilatations = dr.zeros(Float, transient_images[0].height * len(transient_images))
    for i in range(len(transient_images)):
        dr.scatter(datos, transient_images[i].tensor.data, dr.arange(Int, i * transient_images[0].height * transient_images[0].width, (i + 1) * transient_images[0].height * transient_images[0].width)) 
        dr.scatter(wallCameraDilatations, transient_images[i].wallCameraDilation, dr.arange(Int, i * transient_images[0].height, (i + 1) * transient_images[0].height))
    wallCameraDilatations = dr.tile(wallCameraDilatations, numVoxeles)
    print(f"Datos de las imágenes almacenados")

    # Definir vector de lectura en datos
    indices = dr.arange(Int, 0, len(transient_images))
    indices = indices * (transient_images[0].width * transient_images[0].height)

    indices = dr.repeat(indices, len(transient_images))
    indices = dr.tile(indices, numVoxeles)

    voxels = np.zeros((resolution * resolution * resolution, 3), dtype=np.float32)
    i = 0

    for z in range(resolution):
        for y in range(resolution):
            for x in range(resolution):
                fx = bounds.xi + ((x + 0.5) / resolution) * bounds.sx
                fy = bounds.yi + ((y + 0.5) / resolution) * bounds.sy
                fz = bounds.zi + ((z + 0.5) / resolution) * bounds.sz
                
                voxels[i] = np.array([fx, fy, fz])
                i += 1

    voxelesDr = Array3f(voxels)

    print(f"Voxeles generados")
    
    intensidades = sumTransientIntensitiesForOptim(voxelesDr, transient_images, wallPoints, datos, wallCameraDilatations, indices, resolution)

    i = 0
    for z in range(resolution):
        for y in range(resolution):
            for x in range(resolution):

                results[y,x,z] = intensidades[i]
                i += 1

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