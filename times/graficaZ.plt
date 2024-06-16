set terminal pngcairo size 800,600 enhanced font 'Verdana,12'
set output 'tiemposZ.png'
set title 'Tiempos de ejecución para la escena Z'
set xlabel 'Resolución reconstrucción'
set ylabel 'Tiempo en segundos (logscale)'
set logscale y

plot 'Z.txt' using 1:2 with lines title 'Tiempo de ejecución'

