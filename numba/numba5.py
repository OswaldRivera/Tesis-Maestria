import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit

# Parámetros de la simulación
nx, ny = 256, 256   # tamaño de la malla
dx = dy = 1.0
D = 0.1             # coeficiente de difusión
dt = 0.1            # paso de tiempo
steps = 200         # pasos de simulación

# Estado inicial: un "punto caliente" en el centro
u = np.zeros((nx, ny), dtype=np.float32)
u[nx//2, ny//2] = 100.0

@cuda.jit
def diffusion_step_gpu(u, u_new, D, dx, dy, dt):
    i, j = cuda.grid(2)   # índice (x,y) global del hilo
    nx, ny = u.shape
    if 1 <= i < nx-1 and 1 <= j < ny-1:
        u_new[i,j] = u[i,j] + D*dt*(
            (u[i+1,j] - 2*u[i,j] + u[i-1,j])/(dx*dx) +
            (u[i,j+1] - 2*u[i,j] + u[i,j-1])/(dy*dy)
        )
        
# Configuración de bloques e hilos (grid 2D)
threads = (16, 16)
blocks = ((nx + threads[0] - 1) // threads[0],
          (ny + threads[1] - 1) // threads[1])

# Copiar datos a GPU
d_u = cuda.to_device(u)
d_u_new = cuda.device_array_like(d_u)

# Simulación en GPU
for t in range(steps):
    diffusion_step_gpu[blocks, threads](d_u, d_u_new, D, dx, dy, dt)
    d_u, d_u_new = d_u_new, d_u  # swap de arrays

# Traer resultado de GPU a CPU
u_gpu = d_u.copy_to_host()

plt.imshow(u_gpu, cmap="hot")
plt.colorbar()
plt.title("Difusión 2D en GPU con Numba")
plt.show()