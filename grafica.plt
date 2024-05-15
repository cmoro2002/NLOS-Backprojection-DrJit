set terminal pngcairo size 800,600 enhanced font 'Verdana,12'
set output 'tiempos.png'
set title 'Comparación de Tiempos de Ejecución'
set xlabel 'Resolución'
set ylabel 'Tiempo (segundos)'
set logscale y

plot 'numpy.txt' using 1:2 with lines title 'Tiempos numpy (version 2)', \
     'drjit.txt' using 1:2 with lines title 'Tiempos Drjit (version 3)'
