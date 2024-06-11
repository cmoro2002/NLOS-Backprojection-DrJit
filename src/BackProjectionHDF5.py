# Imports de librerias
import time
import argparse
import os
import numpy as np
import drjit as dr

# Imports de drjit
from drjit.llvm import Float, Int, Array3f, UInt32, Loop, UInt, TensorXf

from typing import List
import matplotlib.pyplot as plt

# Imports de mis clases
from TransientImage import TransientImage
from TransientVoxelization import initTransientImage, parseArgsIntoParams
from BoxBounds import BoxBounds
from TransientVoxelizationParams import TransientVoxelizationParams
from FilterResults import apply_filters
from HDF5Reader import parseHDF5
from BackProjectionParams import BackProjectionParams

def visualizarResultado(results, resolution: int, params: TransientVoxelizationParams):
    # Guardar los resultados en un fichero:
    if params.verbose: print(f"Guardando resultados en results/{results}")
    FilterResults = apply_filters(resolution, results)
    # FilterResults = np.squeeze(results[:][:][resolution // 2])

    # Invertir la imagen en el eje vertical
    flipped_results = np.flipud(FilterResults.max_result)

    # Visualizar los resultados
    plt.imshow(flipped_results, cmap='hot', interpolation='nearest')

    plt.colorbar()
    # Guardar la imagen en results/params.resultsRoute
    plt.savefig('results/' + params.resultsRoute + '.png') 
    plt.show()
    
    if params.verbose: print(f"Proceso de backprojection finalizado")

def almacenarResultados( intensidades: Float, resolution: int):
    results = np.zeros((resolution, resolution, resolution))
    i = 0
    for z in range(resolution):
    # z = resolution // 2
        for y in range(resolution):
            for x in range(resolution):

                results[y,x,z] = intensidades[i]
                # results[x,y,z] = intensidades[i]
                i += 1
    return results

def calcularIndices(x: Int, y: Int, width: int) -> Int:
    return (y * width) + x
    
def sumTransientIntensitiesForOptim(voxeles: Array3f, wallPoints: Array3f, r4: Float, numVoxeles: int, BPparams: BackProjectionParams) -> Float:

    altura = BPparams.height

    # Obtener las alturas y la distancia láser-voxel
    alturas = dr.arange(Int,0, altura * BPparams.depth)
    voxelesR = dr.repeat(voxeles, BPparams.depth * altura)

    # r2 (128 distancias)
    if BPparams.confocal:
        # Calcular las distancias voxel-pared para todas las imágenes y alturas
        # Si es confocal r3 = r2, por lo tanto times = r3 + r3 = r3 * 2
        r3 = dr.norm(voxelesR - wallPoints)
        times = r3 * 2
    else:
        # Calcular las distancias voxel-pared para todas las imágenes y alturas
        r3 = dr.norm(voxelesR - wallPoints)
        r2 = dr.norm(voxeles - BPparams.laserWallPos)
        r2 = dr.repeat(r2, altura * BPparams.depth)
        times = r2 + r3

    # Calcular los tiempos y sumar las intensidades
    if BPparams.confocal:
        # Si es confocal r1 = r4
        r1 = r4 
    else:
        r1 = BPparams.r1

    x = Int((times + r1 + r4) / BPparams.t_delta)
    x = dr.clip(x, 0, BPparams.width - 1)

    alturas = dr.tile(alturas, numVoxeles)
 
    indices = calcularIndices(x, alturas, BPparams.width)

    intensities = dr.gather(Float, BPparams.data, indices)

    tam = BPparams.depth * altura

    return dr.block_sum(intensities, tam)

def calcularVoxelesHDF5(voxeles: Array3f, numVoxeles: int, BPparams: BackProjectionParams) -> Float:

    # Calcular los wallpoints de cada imagen 

    wallPoints = dr.tile(BPparams.wallPoints, numVoxeles)
    r4 = dr.tile(BPparams.r4, numVoxeles)
    # print("El número de indices a calcular es de: ", numVoxeles * BPparams.depth * BPparams.height)
    
    return sumTransientIntensitiesForOptim(voxeles, wallPoints, r4, numVoxeles, BPparams)

def generate_voxel_coordinates(volume_position, volume_size, resolution):
    voxel_size = volume_size / resolution
    half_voxel_size = voxel_size / 2.0  # Desplazamiento para centrar el voxel
    voxels = np.zeros((resolution * resolution * resolution, 3), dtype=np.float32)
    i = 0

    for z in range(resolution):
        for y in range(resolution):
            for x in range(resolution):
                fx = volume_position[0] + (x * voxel_size) + half_voxel_size
                fy = volume_position[1] + (y * voxel_size) + half_voxel_size
                fz = volume_position[2] + (z * voxel_size) + half_voxel_size
                
                voxels[i] = np.array([fx, fy, fz])
                i += 1
    
    # print(f"Voxeles generados y centrados")
    return Array3f(voxels)

def backprojectionHDF5(params: TransientVoxelizationParams):
    # Backprojection a partir de un dataset HDF5
    print(f"Empezando el proceso de backprojection para el dataset {params.dataset}")
    BPparams = parseHDF5(params.dataset)

    resolution = params.VOXEL_RESOLUTION
    numVoxeles = resolution * resolution * resolution
    voxelesDr = generate_voxel_coordinates(BPparams.hiddenVolumePosition, BPparams.hiddenVolumeSize, resolution)

    start_time = time.time()
    # limite = 64 * 64 * 32
    # El limite es resolucion al cubo * alto de la imagen * profundidad de la imagen / 2^32
    limite = 2**31

    if (numVoxeles * BPparams.depth * BPparams.height < limite):
        if params.verbose: print(f"Calculando intensidades sin dividir en trozos")
        intensidades = calcularVoxelesHDF5(voxelesDr, numVoxeles, BPparams)
    else:
        intensidades = dr.zeros(Float, numVoxeles)
        numTrozos = numVoxeles * BPparams.depth * BPparams.height // limite
        if params.verbose: print(f"Dividiendo el cálculo en {numTrozos} trozos")
        # Hacer el calculo de intensidades por partes, ya que no se puede hacer con resolucion >= 64
        for i in range(numTrozos):
            limiteReal = limite // BPparams.depth // BPparams.height
            # Si es el último trozo, calcular el resto de voxels
            if (i == numTrozos - 1):
                indicesVoxeles = dr.arange(Int, i * limiteReal, numVoxeles)
                voxelesTrozo = dr.gather(Array3f, voxelesDr, indicesVoxeles)
                numVoxelesTrozo = numVoxeles - i * limiteReal
                dr.scatter( intensidades, calcularVoxelesHDF5(voxelesTrozo, numVoxelesTrozo, BPparams), indicesVoxeles)
            else:
                indicesVoxeles = dr.arange(Int, i * limiteReal, (i + 1) * limiteReal)
                voxelesTrozo = dr.gather(Array3f, voxelesDr, indicesVoxeles)
                dr.scatter( intensidades, calcularVoxelesHDF5(voxelesTrozo, limiteReal,BPparams), indicesVoxeles)

    results = almacenarResultados(intensidades, resolution)

    # Crear matrices de coordenadas voxel    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"El proceso de backprojection ha tardado {elapsed_time} segundos")

    visualizarResultado(results, resolution, params)

    return results
