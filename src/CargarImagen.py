import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

results = np.load(f"letter_ht_90_results.npy")

flattened_results = results[:,:,1]
plt.imshow(flattened_results, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()