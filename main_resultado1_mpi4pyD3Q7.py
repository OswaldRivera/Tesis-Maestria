from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------- Parámetros globales ----------------
tiempo = 100
Nx, Ny, Nz = 50, 50, 53
Lx, Ly, Lz = 20.0, 20.0, 20.0
dx = Lx / Nx
dt = 0.02

# ---------------- División flexible de Nz ----------------
chunk_sizes = [Nz // size + (1 if i < Nz % size else 0) for i in range(size)]
starts = np.cumsum([0] + chunk_sizes[:-1])
Nz_local = chunk_sizes[rank]
z_start = starts[rank]
z_end = z_start + Nz_local

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z_local = np.linspace(-Lz/2 + z_start*Lz/Nz, -Lz/2 + z_end*Lz/Nz, Nz_local)
X, Y, Z = np.meshgrid(x, y, z_local, indexing='ij')

# ---------------- Parámetros físicos ----------------
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
e = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.int32)
w = np.array([1/4,1/8,1/8,1/8,1/8,1/8,1/8], dtype=np.float32)
b = len(e)

# ---------------- Inicialización ----------------
F = np.zeros((b, Nx, Ny, Nz_local), dtype=np.complex64)
A = (p * (X + 1j * Y)).astype(np.complex64)
for i in range(b):
    if i == 0:
        F[i] = (1/(tau*c2))*((beta*D)/(b*c*c))*A
    else:
        F[i] = A*(1 - (dt/(tau*c2))*((beta*D)/(b*c*c)))

# ---------------- Funciones colisión y streaming ----------------
def collision(F, A):
    F_post = np.empty_like(F)
    for i in range(b):
        Feq = (1/(tau*c2))*((beta*D)/(b*c*c))*A if i==0 else A*(1-(dt/(tau*c2))*((beta*D)/(b*c*c)))
        F_post[i] = F[i] - (dt/tau)*(F[i]-Feq) + dt*w[i]*(a*A - d*np.abs(A)**2*A)
    return F_post

def streaming(F_post):
    F_stream = np.empty_like(F_post)
    for i in range(b):
        ex, ey, ez = e[i]
        tmp = np.roll(F_post[i], shift=ex, axis=0)
        tmp = np.roll(tmp, shift=ey, axis=1)
        tmp = np.roll(tmp, shift=ez, axis=2)
        F_stream[i] = tmp
    return F_stream

# ---------------- Bucle temporal ----------------
t1 = time.time()
for t in range(tiempo):
    t0 = time.time()
    A[:] = np.sum(F, axis=0)
    F_post = collision(F, A)
    F_stream = streaming(F_post)

    # Comunicación de halos en Z
    for i in range(b):
        send_top = np.ascontiguousarray(F_stream[i][:,:,-1])
        send_bottom = np.ascontiguousarray(F_stream[i][:,:,0])
        recv_bottom = np.empty((Nx,Ny), dtype=np.complex64)
        recv_top = np.empty((Nx,Ny), dtype=np.complex64)

        req1 = comm.Isend(send_top, dest=(rank+1)%size, tag=100+i)
        req2 = comm.Irecv(recv_bottom, source=(rank-1)%size, tag=100+i)
        req3 = comm.Isend(send_bottom, dest=(rank-1)%size, tag=200+i)
        req4 = comm.Irecv(recv_top, source=(rank+1)%size, tag=200+i)
        MPI.Request.Waitall([req1,req2,req3,req4])

        F_stream[i][:,:,0] = recv_top
        F_stream[i][:,:,-1] = recv_bottom

    F[:] = F_stream
    A[:] = np.sum(F, axis=0)

    if t%10==0 or t==tiempo-1:
        local_max = np.max(np.abs(A))
        local_mean = np.mean(np.abs(A))
        global_max = comm.allreduce(local_max, op=MPI.MAX)
        global_mean = comm.allreduce(local_mean, op=MPI.SUM)/size
        if rank==0:
            print(f"[t={t}] max|A|={global_max:.4e}, mean|A|={global_mean:.4e}, paso_time={time.time()-t0:.4f}s")

t_total = time.time() - t1
if rank==0:
    print(f"Tiempo total de simulación = {t_total:.4f} s")

# ---------------- Reunir resultados con Gatherv ----------------
A_local = np.sum(F, axis=0)
recvcounts = [Nx*Ny*c for c in chunk_sizes] if rank==0 else None
displs = [Nx*Ny*s for s in starts] if rank==0 else None
A_recv = np.empty((size,Nx,Ny), dtype=object) if rank==0 else None

comm.Gatherv(sendbuf=A_local, recvbuf=(A_recv,(recvcounts,displs)), root=0)

if rank==0:
    A_global = np.empty((Nx,Ny,Nz), dtype=np.complex64)
    for r in range(size):
        z0 = starts[r]
        z1 = z0 + chunk_sizes[r]
        A_global[:,:,z0:z1] = A_recv[r]

    os.makedirs("resultados_mpi", exist_ok=True)

    # 1. Re(A) en z=0
    plt.figure(figsize=(6,5))
    plt.imshow(A_global[:,:,Nz//2].real, extent=(-Lx/2,Lx/2,-Ly/2,Ly/2), origin='lower', cmap='RdBu')
    plt.colorbar(label='Re(A)')
    plt.title("Re(A) en z=0")
    plt.savefig("resultados_mpi/ReA.png")

    # 2. |A| sobre x (y=0, z=0)
    plt.figure(figsize=(6,4))
    plt.plot(x, np.abs(A_global[:, Ny//2, Nz//2]))
    plt.title("|A| sobre x (y=0, z=0)")
    plt.xlabel("x")
    plt.ylabel("|A|")
    plt.grid()
    plt.tight_layout()
    plt.savefig("resultados_mpi/AbsA.png")

    # 3. Superficie 3D Re(A) z=0
    ReA_z0 = A_global[:, :, Nz//2].real
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
    plt.savefig("resultados_mpi/2DReA.png")

    # 4. Scatter 3D |A| con máscara
    values = np.abs(A_global)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    mask = (X < Y)
    scatter = ax.scatter(X[mask], Y[mask], Z[mask], c=values[mask], cmap='RdBu', linewidth=0, antialiased=True)
    fig.colorbar(scatter, ax=ax)
    ax.set_title("|A| (mask X<Y)")
    plt.tight_layout()
    plt.savefig("resultados_mpi/3DAbsA_mask.png")

    # 5. Scatter 3D |A| con valores filtrados
    values = np.abs(A_global)
    values[values>=1.1] = np.nan
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, Z, c=values, cmap='Reds', linewidth=0, antialiased=True)
    ax.set_title("|A| filtrado")
    fig.colorbar(scatter, ax=ax)
    plt.tight_layout()
    plt.savefig("resultados_mpi/3DAbsA.png")

    # 6. Scatter 3D Re(A)
    values = np.real(A_global)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, Z, c=values, cmap='RdBu', linewidth=0, antialiased=True)
    ax.set_title("Re(A)")
    fig.colorbar(scatter, ax=ax)
    plt.tight_layout()
    plt.savefig("resultados_mpi/3DReA.png")
    plt.show()
