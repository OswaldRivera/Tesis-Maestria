import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit

# Parámetros de la simulación
nx, ny = 500, 500   # tamaño de la malla
dx = dy = 1.0
D = 1.0             # coeficiente de difusión
dt = 0.2            # paso de tiempo
steps = 500         # pasos de simulación

# Estado inicial: un "punto caliente" en el centro
u = np.zeros((nx, ny), dtype=np.float32)
u[nx//2, ny//2] = 10.0

@jit(nopython=True)
def diffusion_step_cpu(u, D, dx, dy, dt):
    nx, ny = u.shape
    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i,j] = u[i,j] + D*dt*(
                (u[i+1,j] - 2*u[i,j] + u[i-1,j])/(dx*dx) +
                (u[i,j+1] - 2*u[i,j] + u[i,j-1])/(dy*dy)
            )
    return u_new

# Probar CPU
u_cpu = u.copy()
for t in range(50):
    u_cpu = diffusion_step_cpu(u_cpu, D, dx, dy, dt)

plt.imshow(u_cpu, cmap="hot")
plt.colorbar()
plt.title("Difusión 2D en CPU con Numba")
plt.show()
