import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del dominio
tiempo = 100
Nx, Ny, Nz = 50, 50, 50
Lx, Ly, Lz = 20.0, 20.0, 20.0
dx = Lx / Nx # 0.4
dt = 0.02
D = 3
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
z = np.linspace(-Lz / 2, Lz / 2, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Parámetros físicos y CGLE
p = 0.1
c = 10
#tau = 0.5 + np.sqrt(6)/6
tau = 5
D0 = 0.032
c1 = -1.35
c2 = 0.34
beta = D0 * (1 + 1j * c1)
a = 1
d = 1 + 1j * c2

# Red D3Q7 (1 centro + 6 direcciones)
e = np.array([[0, 0, 0],
              [1, 0, 0], [-1, 0, 0],
              [0, 1, 0], [0, -1, 0],
              [0, 0, 1], [0, 0, -1]])

w = np.array([1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])

b = len(e)

# Inicialización de funciones de distribución complejas
F = np.zeros((b, Nx, Ny, Nz), dtype=np.complex128)
Feq = np.zeros_like(F)

# Campo complejo inicial A(x, y, z)
A = p * (X + 1j * Y)
#A = 0.1 * (np.random.rand(Nx, Ny, Nz) + 1j * np.random.rand(Nx, Ny, Nz))

# Distribución de equilibrio inicial

# λ = 1 / (ε c2) = 1 / (tau c2)

for i in range(b):
    if i == 0:
      Feq[i] = (1 / (tau * c2)) * ((beta * D) / (b * c * c)) * A
    else:
      Feq[i] = A * ( 1 - (dt/(tau * c2)) * ((beta * D) / (b * c * c)))

for i in range(b):
    Feq[i][0, :, :]   = Feq[i][1, :, :]   # frontera izquierda copia interior
    Feq[i][-1, :, :]  = Feq[i][-2, :, :]  # frontera derecha copia interior
    # eje y
    Feq[i][:, 0, :]   = Feq[i][:, 1, :]   # frontera arriba
    Feq[i][:, -1, :]  = Feq[i][:, -2, :]  # frontera abajo
    # eje z
    Feq[i][:, :, 0]   = Feq[i][:, :, 1]   # frontera frente
    Feq[i][:, :, -1]  = Feq[i][:, :, -2]  # frontera fondo

F = Feq.copy()

print("Comenzando evolución. Parámetros: Nx,Ny,Nz =", Nx, Ny, Nz, "; dt =", dt, "; tau =", tau)

# Función H(A)
def H(A):
    return a * A - d * np.abs(A)**2 * A

# Evolución temporal
for t in range(tiempo):
    # Macroscópica A
    A = np.sum(F, axis=0)

    # Recalcular equilibrio
    for i in range(b):
      if i == 0:
        Feq[i] = (1 / (tau * c2)) * ((beta * D) / (b * c * c)) * A
      else:
        Feq[i] = A * ( 1 - (1/(tau * c2)) * ((beta * D) / (b * c * c)))

    # Colisión + fuente
    for i in range(b):
        F[i] = F[i] - (dt / tau) * (F[i] - Feq[i]) + dt * w[i] * H(A)

    for i in range(b):
        Feq[i][0, :, :]   = Feq[i][1, :, :]   # frontera izquierda copia interior
        Feq[i][-1, :, :]  = Feq[i][-2, :, :]  # frontera derecha copia interior
        # eje y
        Feq[i][:, 0, :]   = Feq[i][:, 1, :]   # frontera arriba
        Feq[i][:, -1, :]  = Feq[i][:, -2, :]  # frontera abajo
        # eje z
        Feq[i][:, :, 0]   = Feq[i][:, :, 1]   # frontera frente
        Feq[i][:, :, -1]  = Feq[i][:, :, -2]  # frontera fondo
      
    # Diagnóstico rápido para detectar blow-up
    if (t % 10) == 0 or t == tiempo-1:
        maxA = np.max(np.abs(A))
        meanA = np.mean(np.abs(A))
        print(f"t = {t:4d}, max|A| = {maxA:.4e}, mean|A| = {meanA:.4e}")

# Guardar resultado final
A_final = np.sum(F, axis=0)

#######################################################################################
# Graficar Re(A) en z = 0
plt.figure(figsize=(6, 5))
plt.imshow(A_final[:, :, Nz // 2].real, extent=(-Lx/2, Lx/2, -Ly/2, Ly/2), cmap='RdBu')
plt.colorbar(label='Re(A)')
plt.title("Re(A) en z = 0, t = 100 (LBM complejo)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

######################################################################################
# Perfil de densidad |A| en y = 0, z = 0
plt.figure(figsize=(6, 4))
plt.plot(x, np.abs(A_final[:, Ny // 2, Nz // 2]))
plt.title("|A| sobre x (y = 0, z = 0)")
plt.xlabel("x")
plt.ylabel("|A|")
plt.grid()
plt.tight_layout()

#######################################################################################
# Extraer Re(A) en z = 0
ReA_z0 = A_final[:, :, Nz // 2].real

# Crear figura y ejes 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Crear malla de coordenadas 2D para z = 0
X2D, Y2D = np.meshgrid(x, y, indexing='ij')

# Graficar la superficie
surf = ax.plot_surface(X2D, Y2D, ReA_z0, cmap='RdBu', linewidth=0, antialiased=True)

# Etiquetas
ax.set_title("Superficie 3D: Re(A) en z = 0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Re(A)")

# Barra de color
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()

########################################################################################
values = np.abs(A_final)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("|A|")
mask = (X < Y)
scatter = ax.scatter(X[mask], Y[mask], Z[mask], c=values[mask], cmap='RdBu', linewidth=0, antialiased=True)
#scatter = ax.scatter(X, Y, Z, c=values, cmap='RdBu', linewidth=0, antialiased=True)
fig.colorbar(scatter, ax=ax)

plt.tight_layout()

########################################################################################
values = np.real(A_final)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Real(A)")

mask = (X < Y)
#scatter = ax.scatter(X[mask], Y[mask], Z[mask], c=values[mask], cmap='RdBu', linewidth=0, antialiased=True)
scatter = ax.scatter(X, Y, Z, c=values, cmap='RdBu', linewidth=0, antialiased=True)
fig.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()