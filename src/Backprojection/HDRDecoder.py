import imageio

from TransientImage import TransientImage

def decodeHDRFile(file_name: str):

    # Leer la imagen HDR
    img = imageio.imread(file_name, format='HDR')

    # Definir los parametros de la imagen
    width = img.shape[1]
    height = img.shape[0]
    channels = img.shape[2] # Debería ser 3

    max = img.max()
    min = img.min()

    # print(f"max: {max}, min: {min}")

    timeScale = 0.1
    intensityUnit = 0.1

    transientImage = TransientImage(width, height, channels, timeScale, intensityUnit, img, max, min)

    return transientImage