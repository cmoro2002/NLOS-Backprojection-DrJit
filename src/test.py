# python3 src/test.py
import h5py
import numpy as np
from TransientVoxelizationParams import TransientVoxelizationParams
import matplotlib.pyplot as plt
from BackProjectionParams import BackProjectionParams
import drjit as dr

# Imports de drjit
from drjit.llvm import Float, Int, Array3f, UInt32, Loop, UInt, TensorXf



f = h5py.File("datasets/Z.hdf5", 'r')
print(f.keys())
t_delta = np.array(f["deltaT"]).item()
print(t_delta)


t0 = np.array(f["t0"]).item()
print(t0)
data = np.array(f["data"])

# # Compute the mean across dimensions 1 (index 0) and 3 (index 2)
data = np.squeeze(data.mean(axis=(0, 2))) # (4048, 256, 256)

transposed_data = data.transpose(2, 1, 0)
print(transposed_data.shape)
res = transposed_data.reshape(-1)
print("Res")
print(len(res))
# Print the first 4048 values of the dataset
for i in range(4048):
    print(f'Value at position {i}: {res[i]}')

# Select a specific z coordinate, for example, the first one
z_coordinate = 0
heatmap_data = data[:, z_coordinate, :].T

# Print all values of the heatmap

# for i in range(heatmap_data.shape[0]):
#     for j in range(heatmap_data.shape[1]):
#         print(f'Value at position ({i}, {j}): {heatmap_data[i, j]}')

# Plot the heatmap
plt.imshow(heatmap_data, cmap='viridis')
plt.colorbar()
plt.title(f'Heatmap for z coordinate {z_coordinate}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

def parseHDF5(params: TransientVoxelizationParams) -> BackProjectionParams:
    # Leer el dataset
    f = h5py.File(params.dataset, 'r')

    data = np.array(f["data"])

    # Compute the mean across dimensions 1 (index 0) and 3 (index 2)
    data = np.squeeze(data.mean(axis=(0, 2))) # (4048, 256, 256)

    camera = Array3f(np.array(f["cameraPosition"]).T)
    laserOrigin = Array3f(np.array(f["laserPosition"]).flatten())
    laserWallPos = Array3f(np.array(f["laserGridPositions"]).flatten())
    t0 = np.array(f["t0"]).item()
    t_delta = np.array(f["deltaT"]).item()

    wallPoints = np.array(f["cameraGridPositions"])
    wallPoints = wallPoints.reshape(wallPoints.shape[0], -1)
    wallPoints = wallPoints.T

    wallPointsDr = Array3f(wallPoints)

    r1 = dr.norm(laserWallPos - laserOrigin)
    r4 = dr.norm(camera - wallPointsDr)

    BPparams = BackProjectionParams(t0, t_delta, data, r1, r4, wallPointsDr)
    print(BPparams.to_string())
    return BPparams



