import numpy as np
from scipy.signal import convolve

class FilterResults:
    def __init__(self, max_result, max_result_depth, value):
        self.max_result = max_result
        self.max_result_depth = max_result_depth
        self.value = value

def apply_filters(resolution, backprojected):
    max_result = np.zeros((resolution, resolution))
    max_result_depth = np.zeros((resolution, resolution), dtype=int)
    max_intensity = 0

    # # DOUBLE DERIVATIVE Z FILTER
    filtered = np.zeros((resolution, resolution, resolution))

    # LAPLACIAN FILTER
    for x in range(1, resolution - 1):
        for y in range(1, resolution - 1):
            for z in range(1, resolution - 1):
                c = backprojected[x, y, z]
                
                # Suma de vecinos directamente calculada
                neighbours_sum = (
                    backprojected[x-1:x+2, y-1:y+2, z-1:z+2].sum() - c
                )
                
                # Aplicación del filtro
                c = 26 * c - neighbours_sum
                
                # Actualización del máximo
                if c > max_intensity:
                    max_intensity = c
                
                # Almacenamiento del resultado filtrado
                filtered[x, y, z] = c

    # print("Filtro Laplaciano aplicado, normalizando...")
    # Normalize to Final Storage
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                val = filtered[x, y, z] / max_intensity
                if val > 1:
                    val = 1
                backprojected[x, y, z] = val
    
    # print("Valores normalizados, aplicando filtro de máximos...")

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                if max_result[x, y] < backprojected[x, y, z]:
                    max_result[x, y] = backprojected[x, y, z]
                    max_result_depth[x, y] = z

    return FilterResults(max_result, max_result_depth, 1)
