set terminal pngcairo size 800,600 enhanced font 'Verdana,10'
set output 'ejecucion.png'

set title 'Tiempo de ejecución del programa para diferentes resoluciones de entrada R'
set xlabel 'R'
set ylabel 'Tiempo (segundos)'

set grid
set style data linespoints

plot 'logs/Ejecucion.txt' using 1:2 title 'Tiempo de ejecución' with linespoints lt rgb 'blue' lw 2 pt 7
