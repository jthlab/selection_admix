from jax import jit
import numpy as np
import interpax

@jit
def f(x, y):
    return interpax.CubicSpline(x, y)

f(np.array([0, 1, 2, 3]), np.array([0, 1, 4, 9]))
