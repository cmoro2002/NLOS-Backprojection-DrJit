set terminal pngcairo size 800,600 enhanced font 'Verdana,12'
set output 'tiemposBunny.png'
set title 'Tiempos de ejecución para la escena Bunny'
set xlabel 'Resolución reconstrucción'
set ylabel 'Tiempo en segundos (logscale)'
set logscale y

plot 'bunny.txt' using 1:2 with lines title 'Tiempo de ejecución'

