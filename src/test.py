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

data = np.load('datasets/data.npy')

# # Compute the mean across dimensions 1 (index 0) and 3 (index 2)
data = np.squeeze(data.mean(axis=(0, 2))) # (4048, 256, 256)

transposed_data = data.transpose(2, 0, 1)
print(transposed_data.shape)

# Función para plotear grupos de imágenes
def plot_images_in_groups(data, group_size=10):
    num_groups = data.shape[0] // group_size
    for i in range(num_groups):
        fig, axes = plt.subplots(group_size, 1, figsize=(5, group_size * 3))
        fig.suptitle(f'Group {i+1} of Streak Images', fontsize=16)
        for j in range(group_size):
            idx = i * group_size + j
            axes[j].imshow(data[idx], cmap='viridis')
            axes[j].set_title(f'Image {idx}')
            axes[j].axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

# Llamada a la función para plotear las imágenes en grupos de 10
plot_images_in_groups(transposed_data, group_size=10)

# # Select a specific z coordinate, for example, the first one
# z_coordinate = 0
# heatmap_data = data[:, z_coordinate, :].T

# # Plot the heatmap
# plt.imshow(heatmap_data, cmap='viridis')
# plt.colorbar()
# plt.title(f'Heatmap for z coordinate {z_coordinate}')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

# def parseHDF5(params: TransientVoxelizationParams) -> BackProjectionParams:
#     # Leer el dataset
#     f = h5py.File(params.dataset, 'r')

#     data = np.array(f["data"])

#     # Compute the mean across dimensions 1 (index 0) and 3 (index 2)
#     data = np.squeeze(data.mean(axis=(0, 2))) # (4048, 256, 256)

#     camera = Array3f(np.array(f["cameraPosition"]).T)
#     laserOrigin = Array3f(np.array(f["laserPosition"]).flatten())
#     laserWallPos = Array3f(np.array(f["laserGridPositions"]).flatten())
#     t0 = np.array(f["t0"]).item()
#     t_delta = np.array(f["deltaT"]).item()

#     wallPoints = np.array(f["cameraGridPositions"])
#     wallPoints = wallPoints.reshape(wallPoints.shape[0], -1)
#     wallPoints = wallPoints.T

#     wallPointsDr = Array3f(wallPoints)

#     r1 = dr.norm(laserWallPos - laserOrigin)
#     r4 = dr.norm(camera - wallPointsDr)

#     BPparams = BackProjectionParams(t0, t_delta, data, r1, r4, wallPointsDr)
#     print(BPparams.to_string())
#     return BPparams



