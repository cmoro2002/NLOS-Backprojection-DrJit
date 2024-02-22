# Imports de librerias
import time
import argparse
import os
import numpy as np

# Imports de mis clases
from TransientImage import TransientImage
from TransientVoxelization import initTransientImage

def backprojection(transient_images: list):

    z = len(transient_images)

    # TODO: VOXEL_RESOLUTION sería un parametro
    VOXEL_RESOLUTION = 100

    # Para cada x e y de cada imagen 
    for i in range(VOXEL_RESOLUTION):
        for j in range(VOXEL_RESOLUTION):


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
    # Aquí colocas el código principal de tu programa
    print("¡Hola mundo!")

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Descripción de tu programa")
    parser.add_argument("-folder", dest="folder_name", type=str, help="Nombre de la carpeta")

    # Parsear los argumentos de línea de comandos
    args = parser.parse_args()

    # Acceder al valor del argumento -folder
    folder_name = args.folder_name
    print(f"reading files from: {folder_name}")

    # Crear una instancia de TransientImage
    # transient_image = TransientImage(10, 10, 3, 0.1, 0.1, None, 1.0, 0.0)
    transient_images = initTransientImages(folder_name)

    print(f"Empezando el proceso de backprojection para de la carpeta {folder_name}")

    backprojection(transient_images)




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