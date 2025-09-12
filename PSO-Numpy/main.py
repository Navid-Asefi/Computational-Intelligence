import numpy as np
from optimizer import pso_numpy

def sphere(x):
    return np.sum(x**2)

best_pos, best_val = pso_numpy(sphere, dimension=50, bounds=(-100,100))
print("Best position:", best_pos)
print("Best value:", best_val)
