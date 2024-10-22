#!/bin/bash

# Definir las variables
voxelRes=32
scale=64
resultsRoute="Dragon_32"
# dataset="datasets/bunny_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.80,0.53,0.81]_s[256]_l[1]_gs[1.00].hdf5"
# dataset="datasets/Z.hdf5"
dataset="datasets/bunny_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.80,0.53,0.81]_s[256]_l[256]_gs[1.00]_conf.hdf5"
# dataset="datasets/chinese_dragon_l[0.00,-1.00,-0.52]_r[0.00,0.00,3.14]_v[0.75,0.46,0.57]_s[256]_l[256]_gs[1.00]_conf.hdf5"

# Llamar al script Python con las variables como argumentos
python3 src/Backprojection.py \
  -dataset  "$dataset" \
  -voxelRes "$voxelRes" \
  -resultsRoute "$resultsRoute"
