#!/bin/bash

# Definir las variables
voxelRes=32
resultsRoute="Lucy_32"
# dataset="datasets/bunny_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.80,0.53,0.81]_s[256]_l[1]_gs[1.00].hdf5"
dataset="datasets/Z.hdf5"
# dataset="datasets/lucy_l[0.00,-1.00,-0.40]_r[0.00,0.00,1.50]_v[0.44,0.30,0.81]_s[256]_l[1]_gs[1.00].hdf5"

# Llamar al script Python con las variables como argumentos
python3 src/Backprojection.py \
  -dataset  "$dataset" \
  -voxelRes "$voxelRes" \
  -resultsRoute "$resultsRoute"
