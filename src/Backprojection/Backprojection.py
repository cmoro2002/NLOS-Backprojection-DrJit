# Imports de librerias
import time
import argparse
import os
import numpy as np

# Imports de mis clases
from TransientImage import TransientImage
from TransientVoxelization import initTransientImage
from BoxBounds import BoxBounds

def sumTransientIntensitiesFor(fx: float, fy: float, fz: float, transient_images: list):
    voxel = np.array([fx, fy, fz])
    intensities = 0.0

    for transient_image in transient_images:
        for h in range(transient_image.height):
            # Obtener el punto de la pared (no estás utilizando esto en el cálculo de intensidades)
            wall_point = transient_image.getPointForCoord(h)

            # Calcular el tiempo
            time = (np.sqrt(np.sum(np.square(transient_image.getLaser() - voxel))) +
                    np.sqrt(np.sum(np.square(voxel - wall_point))))

            # Sumar la intensidad correspondiente al tiempo
            intensities += transient_image.getIntensityForTime(h, time)

    return intensities


def backprojection(transient_images: list, bounds: BoxBounds):

    num_images = len(transient_images)
    resolution = bounds.resolution

    results = np.zeros((resolution, resolution, resolution))

    for z in range(num_images):
        # Para cada x e y de cada imagen 
        for y in range(resolution):
            for x in range(resolution):

                fx = bounds.xi + ((x + 0.5) / resolution) * bounds.sx
                fy = bounds.yi + ((y + 0.5) / resolution) * bounds.sy
                fz = bounds.zi + ((z + 0.5) / resolution) * bounds.sz

                # Almacenar la suma de los resultados
                results[x, y, z] += sumTransientIntensitiesFor(fx, fy, fz, transient_images)
                 



# Recorrer la lista de imagenes de la carpeta y crear una instancia de TransientImage por cada imagen
def initTransientImages( folder_name: str):

    wall_dir = np.array([1, 0, 0])
    wall_normal = np.array([0, 0, 1])

    # Obtener la lista de archivos en la carpeta
    files = os.listdir(folder_name)

    # Ordenar la lista de archivos en orden alfabético
    files.sort()

    count = 0
    transient_images = []
    
    # Recorrer los archivos y crear una instancia de TransientImage por cada imagen
    for file in files:
        # Comprobar si el archivo es una imagen (puedes ajustar esta comprobación según tus necesidades)
        if file.endswith(('.hdr')):
            file_route = folder_name + "/" + file
            print(f"Reading file: {file_route}")
            transient_image = initTransientImage(file_route)
            
            # Agregar la instancia de TransientImage a la lista
            transient_images.append(transient_image)

            print( f"TransientImage {count}: {transient_image}")
            count += 1

    print(f"Se han leido un total de {count} de imagenes")
    return transient_images



def main():

    # Configuración de la variable de entorno para seleccionar la GPU a utilizar
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Descripción de tu programa")
    parser.add_argument("-folder", dest="folder_name", type=str, help="Nombre de la carpeta")
    parser.add_argument("-voxel_resolution", dest="voxel_resolution", type=int, help="Resolución del voxel")
    parser.add_argument("-max_ortho_size", dest="max_ortho_size", type=int, help="Tamaño máximo del ORTHO")



    # Parsear los argumentos de línea de comandos
    args = parser.parse_args()

    # Acceder al valor del argumento -folder
    folder_name = args.folder_name
    print(f"reading files from: {folder_name}")

    resolucion = args.voxel_resolution
    max_ortho_size = args.max_ortho_size

    # Crear una instancia de TransientImage
    # transient_image = TransientImage(10, 10, 3, 0.1, 0.1, None, 1.0, 0.0)
    transient_images = initTransientImages(folder_name)

    print(f"Empezando el proceso de backprojection para de la carpeta {folder_name}")

    #TODO: Parametros ORTHO_OFFSET x y z
    bounds = BoxBounds(0, 0, 0, max_ortho_size, resolucion)

    backprojection(transient_images, bounds)




if __name__ == "__main__":
    # Registra el tiempo de inicio
    start_time = time.time()

    # Ejecuta el código principal
    main()

    # Registra el tiempo de finalización
    end_time = time.time()

    # Calcula el tiempo total de ejecución
    execution_time = end_time - start_time

    # Muestra el tiempo total de ejecución
    print(f"Tiempo de ejecución: {execution_time} segundos")