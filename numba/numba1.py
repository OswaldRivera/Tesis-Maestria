import numpy as np
import numba as numba
import time

# Python puro
def integrar_trapecio(a, b, N):
    h = (b - a) / N
    s = 0.5 * (np.sin(a) + np.sin(b))
    for i in range(1, N):
        s += np.sin(a + i*h)
    return s * h

# Numba JIT
@numba.jit(nopython=True)
def integrar_trapecio_numba(a, b, N):
    h = (b - a) / N
    s = 0.5 * (np.sin(a) + np.sin(b))
    for i in range(1, N):
        s += np.sin(a + i*h)
    return s * h

N = 1_000_000
t0 = time.time()
res1 = integrar_trapecio(0, np.pi, N)
print("Python puro:", res1, "Tiempo:", time.time() - t0)

t0 = time.time()
res2 = integrar_trapecio_numba(0, np.pi, N)
print("Numba JIT:", res2, "Tiempo:", time.time() - t0)
