set terminal pngcairo size 800,600 enhanced font 'Verdana,12'
set output 'tiempos.png'
set title 'Comparación de Tiempos de Ejecución'
set xlabel 'Resolución reconstrucción'
set ylabel 'Tiempo en segundos (logscale)'
set logscale y

plot 'numpy.txt' using 1:2 with lines title 'Tiempos algoritmo original', \
     'drjit.txt' using 1:2 with lines title 'Tiempos Drjit un voxel (version 3)', \
     'drjitV.txt' using 1:2 with lines title 'Tiempos Drjit todos los voxeles (version 4)'

