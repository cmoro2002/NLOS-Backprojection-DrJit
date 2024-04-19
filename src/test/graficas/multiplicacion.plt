set terminal pngcairo enhanced font 'arial,10'
set output 'graficas/multiplicacion.png'

# Configuración del título y etiquetas de los ejes
set title "Tiempo de ejecución de la multiplicación"
set xlabel "Dimensión"
set ylabel "Tiempo de ejecución (segundos)"

# Configuración de las leyendas
set key outside right top

# Configuración del estilo de las líneas
set style data linespoints

# Grafica 1: Tiempo de ejecución de la multiplicación
plot 'resultados/DrJit.txt' using 1:2 with linespoints title 'Programa DrJit', \
     'resultados/Numpy.txt' using 1:2 with linespoints title 'Programa Numpy'
