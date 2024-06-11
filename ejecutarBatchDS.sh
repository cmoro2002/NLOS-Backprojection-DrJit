#!/bin/bash

# Definir las variables iniciales
folder="src/letter_ht_90"
voxelRes=32
lasers="0.2 0 0"
laser_origin="0.2 0 1"
cam="0.2 0 1"
lookTo="0.2 0 0"
fov=90
t_delta=0.005
ortho="-0.1 -0.35 0.9 0.7"
verbose=False # True or False
resultsRouteBase="batch/HT_64"

# Crear un directorio temporal para las ejecuciones
tempFolder="temp_execution_folder"
mkdir -p $tempFolder

# Crear un directorio para los logs
logsFolder="logs"
mkdir -p $logsFolder

# Número inicial y mínimo de imágenes
numImages=128
minImages=10

# Definir el archivo de log para la salida
logFile="$logsFolder/log.txt"

while [ $numImages -ge $minImages ]
do
    # Crear una copia de la carpeta original
    currentFolder="$tempFolder/letter_ht_$numImages"
    cp -r $folder $currentFolder

    # Eliminar imágenes hasta que queden numImages, empezando por la última
    totalImages=$(ls $currentFolder | wc -l)
    imagesToDelete=$((totalImages - numImages))

    if [ $imagesToDelete -gt 0 ]; then
        ls $currentFolder | sort -r | head -n $imagesToDelete | xargs -I {} rm $currentFolder/{}
    fi

    # Definir el nombre del resultado actual
    currentResultsRoute="${resultsRouteBase}_${numImages}"

    # Ejecutar el script Python y almacenar la salida en un archivo de log
    python3 src/Backprojection.py \
      -manual true \
      -folder "$currentFolder" \
      -voxelRes "$voxelRes" \
      -lasers $lasers \
      -laser_origin $laser_origin \
      -cam $cam \
      -lookTo $lookTo \
      -fov $fov \
      -t_delta $t_delta \
      -ortho $ortho \
      -Optim true \
      -verbose $verbose \
      -resultsRoute "$currentResultsRoute" >> "$logFile" 2>&1

    # Reducir el número de imágenes para la próxima iteración
    numImages=$((numImages - 10))
done

# Eliminar la carpeta temporal
rm -rf $tempFolder
