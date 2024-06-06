import h5py
import numpy as np
from TransientVoxelizationParams import TransientVoxelizationParams
from BackProjectionParams import BackProjectionParams
from Tensor import Tensor2f

import drjit as dr
from drjit.llvm import Array3f, Float, TensorXf, Int

# Reorganizar las coordenadas de (x, z, y) a (x, y, z)
def reorganize_coords(coords):
    return coords[:, [0, 2, 1]] if coords.ndim == 2 else np.array([coords[0], coords[2], coords[1]])

def parseHDF5(dataset) -> BackProjectionParams:
    # Leer el dataset
    f = h5py.File(dataset, 'r')

    data = np.array(f["data"])
    # Guardar la matriz en un archivo de texto
    # np.save('datasets/data.npy', data)

    # data = np.load('datasets/data.npy')
    print(data.shape)
    # data = np.array([0,0,0])

    # Compute the mean across dimensions 1 (index 0) and 3 (index 2)
    data = np.sum(data, axis=2)
    data = np.squeeze(data.mean(axis=(0))) # (4048, 256, 256)
    
    # transposed_data = data.transpose(2, 1, 0)
    transposed_data = data.transpose(1, 2, 0)

    width = transposed_data.shape[2]
    height = transposed_data.shape[1]
    depth = transposed_data.shape[0]

    datos = dr.zeros(Float, height * width * depth)
    print("Shape")
    print(transposed_data[0][:][:].flatten().shape)

    for i in range(depth):
        tensorAux = TensorXf(transposed_data[i][:][:].flatten())
        tensor = Tensor2f(tensorAux.array, (height, width))
        dr.scatter(datos, tensor.data, dr.arange(Int, i * height * width, (i + 1) * height * width)) 

    print(f"Datos de las imágenes almacenados")
    # Aplanar el array a una dimensión, guardarlo en results
    # results = Float(transposed_data.reshape(-1))

    camera_position = np.array(f["cameraPosition"]).T # Trasponer para que pase a filas en vez de columas
    laser_position = np.array(f["laserPosition"]).T
    laser_grid_positions = np.array(f["laserGridPositions"]).flatten()

    # Probar a poner el valor manualmente
    print(f"Laser grid positions: {laser_grid_positions}")
    # laser_grid_positions[1] = -0.5

    t0 = np.array(f["t0"]).item()
    t_delta = np.array(f["deltaT"]).item()
    print(f"Camera: {camera_position}")
    
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

    # print(wallPoints.shape)
    # for i in range (256 * 256 - 10):
    #     print(f"WallPoint en la posición {i}: {wallPoints_reorganized[i]}")

    print(f"WallPoints shape: {wallPoints_reorganized.shape}, wallPoints: {wallPoints_reorganized}")
    print(f"Camera shape: {camera.Shape} , camera: {camera}")
    print(f"LaserOrigin shape: {laserOrigin.Shape}, laserOrigin: {laserOrigin}")
    print(f"LaserWallPos shape: {laserWallPos.Shape}, laserWallPos: {laserWallPos}")

    wallPointsDr = Array3f(wallPoints_reorganized)

    print(f"camera = {camera}")
    print(f"laserOrigin = {laserOrigin}")
    print(f"laserWallPos = {laserWallPos}")
    print(f"wallPointsDr = {wallPointsDr}")

    r1 = dr.norm(laserWallPos - laserOrigin)
    r4 = dr.norm(wallPointsDr - camera)
    print(f"r1 = {r1}")
    print(f"r4 = {r4}")

    hiddenVolumePosition = np.array(f["hiddenVolumePosition"]).flatten()
    hiddenVolumeSize = np.array(f["hiddenVolumeSize"]).item()

    # Ajustar correctamente el volume position para que este arriba a la izquierda y luego calcular hasta size * 2
    hiddenVolumePosition[0] -= (hiddenVolumeSize / 2)
    hiddenVolumePosition[1] -= (hiddenVolumeSize / 2)
    hiddenVolumePosition[2] -= (hiddenVolumeSize / 2)

    # Cambiar coordenadas y a z
    hiddenVolumePosition = np.array([hiddenVolumePosition[0], hiddenVolumePosition[2], hiddenVolumePosition[1]])
    print(f"HiddenVolumePosition: {hiddenVolumePosition}")

    BPparams = BackProjectionParams(laserWallPos, t0, t_delta, width, height, depth, r1, r4, wallPointsDr, hiddenVolumePosition, hiddenVolumeSize, datos)
    print("Primera vez")
    print(BPparams.to_string())
    return BPparams


