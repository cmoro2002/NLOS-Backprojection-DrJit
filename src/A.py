import numpy as np

# Coordenadas de los puntos
P1 = np.array([ 0,  0, -1.4901161e-08])
P2 = np.array([-0.5, 0.0, -0.25])

# Calcular la distancia euclidiana
distancia = np.linalg.norm(P1 - P2)

print(f"La distancia entre los puntos es: {distancia}")
