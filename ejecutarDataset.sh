#!/bin/bash

# Definir las variables
folder="src/letter_ht_90"
voxelRes=32
lasers="0.2 0 0"
laser_origin="0.2 0 1"
cam="0.2 0 1"
lookTo="0.2 0 0"
fov=90
t_delta=0.005
ortho="-0.1 -0.35 0.9 0.7"
resultsRoute="HT_32_B"
dataset="datasets/Z.hdf5"

# Llamar al script Python con las variables como argumentos
python3 src/Backprojection.py \
  -dataset  "$dataset" \
  -folder "$folder" \
  -voxelRes "$voxelRes" \
  -Optim true \
  -resultsRoute "$resultsRoute"