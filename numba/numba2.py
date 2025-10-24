import numpy as np
import numba
import time
import matplotlib.pyplot as plt

@numba.jit(nopython=True)
def euler_logistica(r, y0, t_max, dt):
    n = int(t_max / dt)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        y[i] = y[i-1] + dt * r * y[i-1] * (1 - y[i-1])
    return y

# Parametros
r = 2.0
y0 = 0.1
t_max = 1000.0
dt = 0.001

# Simulacion
y = euler_logistica(r, y0, t_max, dt)

plt.plot(np.linspace(0, t_max, len(y)), y)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Ecuación logística con Euler (Numba)")
plt.show()


