import cv2
import numpy as np

import drjit as dr

from drjit.llvm import Float, TensorXf

from TransientImage import TransientImage

def decodeHDRFile(file_name: str): 

    # Leer la imagen HDR
    img = cv2.imread(file_name, flags=cv2.IMREAD_ANYDEPTH)

    channels = img.shape[2] # Debería ser 3
    
    # Me quedo solo con el canal 0
    img = img[:, :, 0]

    # Definir los parametros de la imagen
    width = img.shape[1]
    height = img.shape[0]

    max = img.max()
    min = img.min()

    timeScale = 0.1
    intensityUnit = 0.1

    transientImage = TransientImage(width, height, channels, timeScale, intensityUnit, img, max, min)

    return transientImage
