import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import prange

# ---------------- Parametros ----------------
tiempo = 240
Nx, Ny, Nz = 50, 50, 50
Lx, Ly, Lz = 20.0, 20.0, 20.0
dx = Lx / Nx
dt = 0.02

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(-Lz/2, Lz/2, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# ---------------- Parametros fisicos ----------------
p = 0.1
tau = 5.0
D = 3.0
c = 10.0
D0 = 0.032
c1 = -1.35
c2 = 0.34
beta = D0 * (1 + 1j * c1)
a = 1.0
d = 1.0 + 1j * c2

# ---------------- Red D3Q7 ----------------
e = np.array([[0, 0, 0],
              [1, 0, 0], [-1, 0, 0],
              [0, 1, 0], [0, -1, 0],
              [0, 0, 1], [0, 0, -1]], dtype=np.int32)
w = np.array([1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8], dtype=np.float32)
b = len(e)

# ---------------- Inicializacion ----------------
F = np.zeros((b, Nx, Ny, Nz), dtype=np.complex64)

r = 2.0
q = 2.0
L = 20.0
A = (p * (X - r * np.cos((q * np.pi * Z)/L) + 1j * (Y - r * np.sin((q * np.pi * Z)/L)))).astype(np.complex64)

@numba.njit(parallel=True)
def streaming(F, e, w, tau, c2, beta, D, b, c, dt, a, d, A):
    Nx, Ny, Nz = F.shape[1:]
    for i in prange(b):
        # Fuerza de equilibrio
        if i == 0:
            Feq_i = (1 / (tau*c2)) * ((beta*D)/(b*c*c)) * A
        else:
            Feq_i = A * (1 - (dt/(tau*c2)) * ((beta*D)/(b*c*c)))
        
        # Colision + fuerza
        F_post = F[i] - (dt/tau)*(F[i]-Feq_i) + dt*w[i]*(a*A - d*np.abs(A)**2*A)
        
        ex, ey, ez = e[i]
        
        # Streaming 
        for x in prange(Nx):
            x_to = (x+ex)%Nx
            for y in prange(Ny):
                y_to = (y+ey)%Ny
                for z in prange(Nz):
                    z_to = (z+ez)%Nz
                    F[i, x_to, y_to, z_to] = F_post[x, y, z]

# ---------------- Inicializacion de equilibrio ----------------
for i in prange(b):
    if i == 0:
        F[i] = (1 / (tau*c2)) * ((beta*D)/(b*c*c)) * A
    else:
        F[i] = A * (1 - (dt/(tau*c2)) * ((beta*D)/(b*c*c)))

# ---------------- Evolucion temporal ----------------
for t in prange(tiempo):
    # Calcular A secuencialmente
    A[:] = 0
    for i in prange(b):
        A += F[i]
    
    # Colision + streaming 
    streaming(F, e, w, tau, c2, beta, D, b, c, dt, a, d, A)
    
    # Frontera
    for i in prange(b):
        F[i][0, :, :]   = F[i][1, :, :]
        F[i][-1, :, :]  = F[i][-2, :, :]
        F[i][:, 0, :]   = F[i][:, 1, :]
        F[i][:, -1, :]  = F[i][:, -2, :]
        F[i][:, :, 0]   = F[i][:, :, 1]
        F[i][:, :, -1]  = F[i][:, :, -2]
    
    if t % 10 == 0 or t == tiempo-1:
        print(f"t={t}, max|A|={np.max(np.abs(A)):.4e}, max|A|={np.min(np.abs(A)):.4e} ")

# ---------------- Resultado final ----------------
A_final = A

# ---------------- Visualizacion ----------------

# 1. Re(A) en z=0
plt.figure(figsize=(6,5))
plt.imshow(A_final[:,:,Nz//2].real, extent=(-Lx/2,Lx/2,-Ly/2,Ly/2), origin='lower', cmap='RdBu')
plt.colorbar(label='Re(A)')
plt.title("Re(A) en z=0")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("resultados_main/ReA.png")

# 2. |A| sobre x (y=0, z=0)
plt.figure(figsize=(6,4))
plt.plot(x, np.abs(A_final[:, Ny//2, Nz//2]))
plt.title("|A| sobre x (y=0, z=0)")
plt.xlabel("x")
plt.ylabel("|A|")
plt.grid()
plt.tight_layout()
plt.savefig("resultados_main/AbsA.png")

# 3. Superficie 3D Re(A) z=0
ReA_z0 = A_final[:, :, Nz//2].real
X2D, Y2D = np.meshgrid(x, y, indexing='ij')
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X2D, Y2D, ReA_z0, cmap='RdBu', linewidth=0, antialiased=True)
ax.set_title("Superficie 3D: Re(A) en z=0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Re(A)")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()
plt.savefig("resultados_main/2DReA.png")

# 4. Scatter 3D |A|
values = np.abs(A_final)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
#mask = (Z <= 0)
#scatter = ax.scatter(X[mask], Y[mask], Z[mask], c=values[mask], cmap='RdBu', linewidth=0, antialiased=True)
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            if values[i][j][k] >= 0.75:
                values[i][j][k] = np.nan
scatter = ax.scatter(X, Y, Z, c=values, cmap='RdBu', linewidth=0, antialiased=True)
ax.set_title("|A|")
fig.colorbar(scatter, ax=ax)
plt.tight_layout()
plt.savefig("resultados_main/3DAbsA.png")

# 5. Scatter 3D Re(A)
values = np.real(A_final)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
mask = (X < Y)
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            if values[i][j][k] <= -0.75 or values[i][j][k] >= 0.75:
              values[i][j][k] = np.nan
scatter = ax.scatter(X[mask], Y[mask], Z[mask], c=values[mask], cmap='RdBu', linewidth=0, antialiased=True)
#scatter = ax.scatter(X, Y, Z, c=values, cmap='RdBu', linewidth=0, antialiased=True)
ax.set_title("Re(A)")
fig.colorbar(scatter, ax=ax)
plt.tight_layout()
plt.savefig("resultados_main/3DReA.png")
plt.show()
