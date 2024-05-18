import h5py
import numpy as np
from TransientVoxelizationParams import TransientVoxelizationParams



f = h5py.File("datasets/Z.hdf5", 'r')
print(f.keys())
data = f["data"]

print(data.shape)


def parseHDF5(parsed_args, params: TransientVoxelizationParams):
    # Leer el dataset
    f = h5py.File(params.dataset, 'r')
    data = f["data"]

    # Leer los parámetros de la voxelización
    params.fov # Preguntar a julio como traducir a fov el dataset
    params.camera = np.array(f["cameraPosition"])
    # params.lookTo
    params.laserOrigin = np.array(f["laserPosition"])
    # params.t0 
    params.t_delta = f["deltaT"]


