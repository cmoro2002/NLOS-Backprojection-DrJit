set title "Comparativa de tiempos de ejecución - Multiplicación"
set xlabel "Dimensión"
set ylabel "Tiempo (s)"
set logscale x
set logscale y
set grid
plot "resultados/DrJit.txt" using 1:2 with lines title "DrJit", \
     "resultados/Numpy.txt" using 1:2 with lines title "NumPy"

pause -1 "Presiona enter para continuar con la siguiente gráfica..."

reset

set title "Comparativa de tiempos de ejecución - Suma de Matrices"
set xlabel "Dimensión"
set ylabel "Tiempo (s)"
set logscale x
set logscale y
set grid
plot "resultados/DrJit.txt" using 1:3 with lines title "DrJit", \
     "resultados/Numpy.txt" using 1:3 with lines title "NumPy"

pause -1 "Presiona enter para continuar con la siguiente gráfica..."

reset

set title "Comparativa de tiempos de ejecución - Suma de Vector"
set xlabel "Dimensión"
set ylabel "Tiempo (s)"
set logscale x
set logscale y
set grid
plot "resultados/DrJit.txt" using 1:4 with lines title "DrJit", \
     "resultados/Numpy.txt" using 1:4 with lines title "NumPy"

pause -1 "Presiona enter para continuar con la siguiente gráfica..."