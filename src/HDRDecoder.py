import cv2
import numpy as np

import drjit as dr

from drjit.llvm import Float, TensorXf

from TransientImage import TransientImage

def decodeHDRFile(file_name: str): 

    # Leer la imagen HDR
    img = cv2.imread(file_name, flags=cv2.IMREAD_ANYDEPTH)

    print(img.shape)

    tensor = TensorXf(img)

    # Definir los parametros de la imagen
    width = img.shape[1]
    height = img.shape[0]
    channels = img.shape[2] # Debería ser 3

    max = img.max()
    min = img.min()

    timeScale = 0.1
    intensityUnit = 0.1

    transientImage = TransientImage(width, height, channels, timeScale, intensityUnit, tensor, max, min)

    return transientImage
