# Imports de librerias
import time

# Imports de mis clases
from TransientImage import TransientImage

def main():
    # Aquí colocas el código principal de tu programa
    print("¡Hola mundo!")

    # Crear una instancia de TransientImage
    transient_image = TransientImage(10, 10, 3, 0.1, 0.1, None, 1.0, 0.0)

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