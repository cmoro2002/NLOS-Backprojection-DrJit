import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
from Tensor import Tensor2f
from BackProjectionParams import BackProjectionParams
from TransientVoxelizationParams import TransientVoxelizationParams
from BackProjectionHDF5 import backprojectionHDF5

# Imports de drjit
from drjit.llvm import Float, Int, Array3f, TensorXf

import mitsuba as mi

# mi.set_variant('cuda_ad_rgb')
mi.set_variant('llvm_ad_rgb')
sys.path.insert(1, '../..')

import mitransient as mitr

# Función de la librería tal de @diegoroyo
def get_grid_xyz(nx, ny, rw_scale_x, rw_scale_y):
    px = rw_scale_x
    py = rw_scale_y
    xg = np.stack(
        (np.linspace(-px, px, num=2*nx + 1)[1::2],)*ny, axis=1)
    yg = np.stack(
        (np.linspace(-py, py, num=2*ny + 1)[1::2],)*nx, axis=0)
    assert xg.shape[0] == yg.shape[0] == nx and xg.shape[1] == yg.shape[1] == ny, \
        'Incorrect shapes'
    return np.stack([xg, yg, np.zeros((nx, ny))], axis=-1).astype(np.float32)



# Variables for the scene

hiddenVolumePos = np.array([0.0, 0.0, 1.0])

hiddenVolumeSize = 0.81

laserOrigin = np.array([-0.5, 0.0, 0.25])
cameraOrigin = np.array([-0.5, 0.0, 0.25])
height = 64
depth = 64
width = 600
deltaT = 0.006
t0 = 1.85
# t0 = 0.0
sampleAmount = 20000
wallPoints = get_grid_xyz(height, depth, 1, 1)
laserWallPos = wallPoints[32,32,:]
wallPointsDr = Array3f(wallPoints.reshape(height * depth, 3))

# Variables in the form of drjit types

laserOriginDr = Array3f(laserOrigin)
cameraOriginDr = Array3f(cameraOrigin)
laserWallPosDr = Array3f(laserWallPos)

# Load the geometry of the hidden scene
geometry = mi.load_dict(
    {
        "type": "obj",
        "filename": "datasets/Z.obj",
        "to_world": mi.ScalarTransform4f.translate(hiddenVolumePos),
        "bsdf": {"type": "diffuse", "reflectance": 1.0},
    }
)

# Load the emitter (laser) of the scene
emitter = mi.load_dict(
    {
        "type": "projector",
        "irradiance": 100.0,
        "fov": 0.2,
        "to_world": mi.ScalarTransform4f.translate(laserOrigin),
    }
)

# Define the transient film which store all the data
transient_film = mi.load_dict(
    {
        "type": "transient_hdr_film",
        "width": depth,
        "height": height,
        "temporal_bins": width,
        "bin_width_opl": deltaT,
        "start_opl": t0,
        "rfilter": {"type": "box"},
    }
)

# Define the sensor of the scene
nlos_sensor = mi.load_dict(
    {
        "type": "nlos_capture_meter",
        "sampler": {"type": "independent", "sample_count": sampleAmount},
        "account_first_and_last_bounces": False,
        "sensor_origin": cameraOrigin,
        "transient_film": transient_film,
    }
)

# Load the relay wall. This includes the custom "nlos_capture_meter" sensor which allows to setup measure points directly on the shape and importance sample paths going through the relay wall.
relay_wall = mi.load_dict(
    {
        "type": "rectangle",
        "bsdf": {"type": "diffuse", "reflectance": 1.0},
        "nlos_sensor": nlos_sensor,
    }
)

# Finally load the integrator
integrator = mi.load_dict(
    {
        "type": "transient_nlos_path",
        "nlos_laser_sampling": True,
        "nlos_hidden_geometry_sampling": True,
        "nlos_hidden_geometry_sampling_do_rroulette": False,
        "temporal_filter": "box",
    }
)

# Assemble the final scene
scene = mi.load_dict({
    'type' : 'scene',
    'geometry' : geometry,
    'emitter' : emitter,
    'relay_wall' : relay_wall,
    'integrator' : integrator
})


# Now we focus the emitter to irradiate one specific pixel of the "relay wall"
pixel = mi.Point2f(32, 32)
mitr.nlos.focus_emitter_at_relay_wall_pixel(pixel, relay_wall, emitter)

# Prepare transient integrator for transient path tracing
# Does the work to initialize the scene before path tracing
transient_integrator = scene.integrator()
transient_integrator.prepare_transient(scene, sensor=0)
# sensor=0 just tells it to use the first sensor in the scene
# (in case there is more than one sensor in the XML file)

# Render the scene and develop the data
data_steady, data_transient = transient_integrator.render(scene)
# And evaluate the output to launch the corresponding kernel
dr.eval(data_steady, data_transient)

# Mean the rgb channels
# data = np.squeeze(data.mean(axis=(0)))
data_transient = np.squeeze(np.array(data_transient).mean(axis=3))
# data_transient = data_transient.transpose(1,0,2)

datos = dr.zeros(Float, height * width * depth)

# Chapuza para asegurarme de que los datos se guardan correctamente (numpy alguna vez no lo ha hecho como yo quería)
for i in range(depth):
    tensorAux = TensorXf(data_transient[i][:][:].flatten())
    tensor = Tensor2f(tensorAux.array, (height, width))
    dr.scatter(datos, tensor.data, dr.arange(Int, i * height * width, (i + 1) * height * width)) 

# r4 = dr.norm(wallPointsDr - cameraOriginDr)
# r1 = dr.norm(laserWallPosDr - laserOriginDr)
r4 = dr.zeros(Float,height * depth)
r1 = 0.0

# Desplazar el hidden volume para que calcular solo valores positivos (hacermelo mas facil en la reconstruccion)
# hiddenVolumePos = np.array([-0.2, 0.0, 1.0])
hiddenVolumePos -= (hiddenVolumeSize / 2)

res = BackProjectionParams(laserWallPosDr, t0, deltaT, width, height, depth, r1, r4, wallPointsDr, hiddenVolumePos, hiddenVolumeSize, datos, False)
print(res.to_string())

params = TransientVoxelizationParams()
params.VOXEL_RESOLUTION = 64
params.resultsRoute = "mitsuba"

backprojectionHDF5(params, res)