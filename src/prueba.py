import numpy as np
import drjit as dr
import os

# from drjit.cuda import Float, UInt32
from drjit.llvm import Float, Int

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Create some floating-point arrays
a = Float([1.0, 2.0, 3.0, 4.0])
b = Float([4.0, 3.0, 2.0, 1.0])

# Perform a loop operation on the GPU
c = dr.loop(a, b, UInt32(4), lambda x, y: x + y)

print(f'c -> ({type(c)}) = {c}')

# Perform simple arithmetic
c = a + 2.0 * b

print(f'c -> ({type(c)}) = {c}')

# Convert to NumPy array
d = np.array(c)

print(f'd -> ({type(d)}) = {d}')

# Initialize floating-point array of size 5 with a constant value
a = dr.full(Float, 0.1, 5) # np.ones(5, 0.4)
print(f'dr.full: {a}')