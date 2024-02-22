from TransientImage import TransientImage
from HDRDecoder import decodeHDRFile
from StreakLaser import StreakLaser

def initTransientImage(file_name: str):

    # TODO: Definir StreakLaser a partir del file
    streakLaser = StreakLaser(0, 0)

    transient_image = decodeHDRFile(file_name)

    # TODO: Definir parametros de la camara

    # TODO: Definir distancia al laser  y su hit time

    return transient_image




