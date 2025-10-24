from numba import vectorize
import numpy as np

@vectorize(['float64(float64, float64)'], nopython=True)
def f(x, y):
    return x**2 + y**2

x = np.arange(1e6)
y = np.arange(1e6)
z = f(x, y)  

print(z)
