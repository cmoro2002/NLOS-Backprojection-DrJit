import h5py
import numpy as np
import os
from TransientVoxelizationParams import TransientVoxelizationParams
from BackProjectionParams import BackProjectionParams
from Tensor import Tensor2f

import drjit as dr
from drjit.llvm import Array3f, Float, TensorXf, Int

# Reorganizar las coordenadas de (x, z, y) a (x, y, z)
def reorganize_coords(coords):
    return coords[:, [0, 2, 1]] if coords.ndim == 2 else np.array([coords[0], coords[2], coords[1]])

def parseHDF5(dataset, scaleDownTo) -> BackProjectionParams:
    # Leer el dataset
    f = h5py.File(dataset, 'r')
    confocal = np.array(f["isConfocal"]).item()

    # npy_path = f'{datasetReal}.npy'
    data = np.array(f["data"])

    # Compute the mean across dimensions 1 (index 0) and 3 (index 2)
    data = np.sum(data, axis=2)
    data = np.squeeze(data.mean(axis=(0))) # (4048, 256, 256)
    # data = np.squeeze(data) # (4048, 256, 256)
    print(f"Data shape = {data.shape}")

    if scaleDownTo != None:
        print(f"Data shape primero = {data.shape}")
        data = downsample_matrix(data, scaleDownTo)
        print(f"Data shape Segundo = {data.shape}")
    
    # transposed_data = data.transpose(2, 1, 0)
    transposed_data = data.transpose(1, 2, 0)

    print("Transposed data shape = ", transposed_data.shape)

    # print(f"Data shape = {transposed_data.shape}")
    # input()

    width = transposed_data.shape[2]
    height = transposed_data.shape[1]
    depth = transposed_data.shape[0]

    datos = dr.zeros(Float, height * width * depth)

    for i in range(depth):
        tensorAux = TensorXf(transposed_data[i][:][:].flatten())
        tensor = Tensor2f(tensorAux.array, (height, width))
        dr.scatter(datos, tensor.data, dr.arange(Int, i * height * width, (i + 1) * height * width)) 

    print(f"datos = {datos}")
    # print(f"Datos de las imágenes almacenados")
    # Aplanar el array a una dimensión, guardarlo en results
    # results = Float(transposed_data.reshape(-1))

    camera_position = np.array(f["cameraPosition"]).T # Trasponer para que pase a filas en vez de columas
    laser_position = np.array(f["laserPosition"]).T
    laser_grid_positions = np.array(f["laserGridPositions"]).flatten()

    t0 = np.array(f["t0"]).item()
    t_delta = np.array(f["deltaT"]).item()
    
    # Aplicar la reorganización a las posiciones
    camera_position_reorganized = reorganize_coords(camera_position)
    laser_position_reorganized = reorganize_coords(laser_position)
    laser_grid_positions_reorganized = reorganize_coords(laser_grid_positions)

    # Convertir a Array3f
    camera = Array3f(camera_position_reorganized)
    laserOrigin = Array3f(laser_position_reorganized)
    laserWallPos = Array3f(laser_grid_positions_reorganized)

    wallPoints = np.array(f["cameraGridPositions"])
    wallPoints = wallPoints.reshape(wallPoints.shape[0], -1)
    wallPoints = wallPoints.T

    # Intercambiar coordenadas y y z
    wallPoints_reorganized = wallPoints[:, [0, 2, 1]]
    wallPointsDr = Array3f(wallPoints_reorganized)

    r4 = dr.norm(wallPointsDr - camera)
    if confocal: 
        r1 = r4
    else:
        r1 = dr.norm(laserWallPos - laserOrigin)


    hiddenVolumePosition = np.array(f["hiddenVolumePosition"]).flatten()
    size = np.array(f["hiddenVolumeSize"])
    print(size)
    hiddenVolumeSize = np.array(f["hiddenVolumeSize"]).item()
    # hiddenVolumeSize = 0.81

    # Ajustar correctamente el volume position para que este arriba a la izquierda y luego calcular hasta size * 2
    hiddenVolumePosition[0] -= (hiddenVolumeSize / 2)
    hiddenVolumePosition[1] -= (hiddenVolumeSize / 2)
    hiddenVolumePosition[2] -= (hiddenVolumeSize / 2)

    # Cambiar coordenadas y a z
    hiddenVolumePosition = np.array([hiddenVolumePosition[0], hiddenVolumePosition[2], hiddenVolumePosition[1]])

    res = BackProjectionParams(laserWallPos, t0, t_delta, width, height, depth, r1, r4, wallPointsDr, hiddenVolumePosition, hiddenVolumeSize, datos, confocal)
    print(res.to_string())
    return res

def downsample_matrix(matrix, scaleDownTo):
    """
    Realiza el downsampling de una matriz utilizando el promedio de bloques.
    
    Args:
    matrix (np.ndarray): La matriz original de tamaño (4048, 256, 256).
    scaleDownTo (int): La nueva dimensión para las dos últimas dimensiones (deben ser divisibles por 256).
    
    Returns:
    np.ndarray: La matriz después del downsampling con tamaño (4048, scaleDownTo, scaleDownTo).
    """
    # Verificar que las dimensiones originales sean divisibles por la nueva dimensión
    assert matrix.shape[1] % scaleDownTo == 0, "La dimensión original debe ser divisible por scaleDownTo."
    assert matrix.shape[2] % scaleDownTo == 0, "La dimensión original debe ser divisible por scaleDownTo."

    # Calcular el factor de escala
    scale_factor = matrix.shape[1] // scaleDownTo

    # Realizar el downsampling usando el promedio de bloques
    downsampled_matrix = matrix.reshape((matrix.shape[0], scaleDownTo, scale_factor, scaleDownTo, scale_factor)).mean(axis=(2, 4))

    return downsampled_matrix