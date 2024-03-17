set terminal pngcairo enhanced font 'arial,10'
set output 'graficas/sumaElementos.png'

# Configuración del título y etiquetas de los ejes
set title "Tiempo de ejecución de la suma vectorial"
set xlabel "Dimensión"
set ylabel "Tiempo de ejecución (segundos)"

# Configuración de las leyendas
set key outside right top

# Configuración del estilo de las líneas
set style data linespoints

# Grafica 3: Tiempo de ejecución de la suma vectorial
plot 'A.txt' using 1:4 with linespoints title 'Programa A', \
     'B.txt' using 1:4 with linespoints title 'Programa B'
