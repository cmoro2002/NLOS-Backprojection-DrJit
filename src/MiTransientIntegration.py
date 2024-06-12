import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
from Tensor import Tensor2f
from BackProjectionParams import BackProjectionParams
from TransientVoxelizationParams import TransientVoxelizationParams
from BackProjectionHDF5 import backprojectionHDF5, almacenarResultados
from FilterResults import apply_filters
from pynput import keyboard 

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

def visualizarResultado(results, resolution: int, params: TransientVoxelizationParams):
    FilterResults = apply_filters(resolution, results)

    # Invertir la imagen en el eje vertical
    flipped_results = np.flipud(FilterResults.max_result)

    # Visualizar los resultados
    plt.clf()
    plt.imshow(flipped_results, cmap='hot', interpolation='nearest')
    plt.colorbar()
    print("Pausing...")
    plt.pause(0.1)


def loadScene(hiddenVolumePos, hiddenVolumeSize, laserOrigin, cameraOrigin, height, depth, width, deltaT, t0, sampleAmount):
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

    return scene

def generateTransientDataFromScene(scene):
    # Prepare transient integrator for transient path tracing
    # Does the work to initialize the scene before path tracing
    transient_integrator = scene.integrator()
    transient_integrator.prepare_transient(scene, sensor=0)

    # Render the scene and develop the data
    data_steady, data_transient = transient_integrator.render(scene)
    # And evaluate the output to launch the corresponding kernel
    dr.eval(data_steady, data_transient)

    return data_transient

def convertDataToArray(data_transient, height, depth, width):
    # Mean the rgb channels
    data_transient = np.squeeze(np.array(data_transient).mean(axis=3))


    # Chapuza para asegurarme de que los datos se guardan correctamente (numpy alguna vez no lo ha hecho como yo quería)
    datos = dr.zeros(Float, height * width * depth)
    for i in range(depth):
        tensorAux = TensorXf(data_transient[i][:][:].flatten())
        tensor = Tensor2f(tensorAux.array, (height, width))
        dr.scatter(datos, tensor.data, dr.arange(Int, i * height * width, (i + 1) * height * width)) 

    return datos

def generateReconstruction(scene, hiddenVolumePos, hiddenVolumeSize, laserWallPosDr, t0, deltaT, width, height, depth, wallPointsDr):
    data_transient = generateTransientDataFromScene(scene)

    dataArray = convertDataToArray(data_transient, height, depth, width)

    r4 = dr.zeros(Float,height * depth)
    r1 = 0.0

    # Move the hiddenVolume coordinates to the corner, for backprojection to work properly
    hiddenVolumePosForReconstruction = hiddenVolumePos - (hiddenVolumeSize / 2)
    print(f"hiddenVolumePosForReconstruction = {hiddenVolumePosForReconstruction}")

    backProjectionParameters = BackProjectionParams(laserWallPosDr, t0, deltaT, width, height, depth, r1, r4, wallPointsDr, hiddenVolumePosForReconstruction, hiddenVolumeSize, dataArray, False)

    return backprojectionHDF5(params, backProjectionParameters)

# Función para manejar las pulsaciones de teclas
def on_press(key):
    print("Pulsadoooo")
    global hiddenVolumePos
    try:
        if key.char == 'w':
            hiddenVolumePos[1] += 0.1  # Mover hacia arriba
        elif key.char == 's':
            hiddenVolumePos[1] -= 0.1  # Mover hacia abajo
        elif key.char == 'a':
            hiddenVolumePos[0] -= 0.1  # Mover hacia la izquierda
        elif key.char == 'd':
            hiddenVolumePos[0] += 0.1  # Mover hacia la derecha
        elif key.char == 'j':
            hiddenVolumePos[2] += 0.1 
        elif key.char == 'k':
            hiddenVolumePos[2] -= 0.1
        elif key.char == 'q':
            exit(0)
    except AttributeError:
        pass

# plt.ion()

# Variables for the scene
hiddenVolumePos = np.array([0.0, -0.2, 1.0])
hiddenVolumeReconstructionPos = np.array([0.0, 0.0, 1.0])
hiddenVolumeSize = 0.91
laserOrigin = np.array([-0.5, 0.0, 0.25])
cameraOrigin = np.array([-0.5, 0.0, 0.25])
height = 64
depth = 64
width = 600
deltaT = 0.006
t0 = 1.85
sampleAmount = 300
voxelResolution = 64
resultsRoute = "mitsuba"
loopAmount = 10

# Define necessary parameters for backprojection
params = TransientVoxelizationParams()
params.VOXEL_RESOLUTION = voxelResolution
params.resultsRoute = resultsRoute

wallPoints = get_grid_xyz(height, depth, 1, 1)
laserWallPos = wallPoints[32,32,:]

# Variables in the form of drjit types
wallPointsDr = Array3f(wallPoints.reshape(height * depth, 3))
laserWallPosDr = Array3f(laserWallPos)

timeInit = time.time()



# intensities = dr.zeros(Float, voxelResolution ** 3)
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Detener el listener al cerrar la ventana de matplotlib
# listener.join()

while(True):
    scene = loadScene(hiddenVolumePos, hiddenVolumeSize, laserOrigin, cameraOrigin, height, depth, width, deltaT, t0, sampleAmount)
    intensities = generateReconstruction(scene, hiddenVolumeReconstructionPos, hiddenVolumeSize, laserWallPosDr, t0, deltaT, width, height, depth, wallPointsDr)

    reconstruction = almacenarResultados(intensities, voxelResolution)

    visualizarResultado(reconstruction, voxelResolution, params)




# print("Fin de la ejecución")