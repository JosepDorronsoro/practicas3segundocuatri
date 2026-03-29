import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import cuda

# --- 1. KERNELS DE CUDA (Ejecutados en la GPU) ---

@cuda.jit
def compute_accelerations_cuda(pos, mass, acc, softening, G):
    # Obtener el índice global del hilo (representa a nuestro cuerpo 'i')
    i = cuda.grid(1)
    n = pos.shape[0]
    
    # Asegurarnos de no calcular fuera de los límites de nuestros datos
    if i < n:
        # Variables locales en los registros rápidos de la GPU
        acc_x = 0.0
        acc_y = 0.0
        acc_z = 0.0
        
        # Cada hilo 'i' itera sobre todos los cuerpos 'j'
        for j in range(n):
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                
                # Distancia al cuadrado + softening
                dist_sq = dx**2 + dy**2 + dz**2 + softening**2
                
                # 1 / dist^3
                inv_dist3 = 1.0 / math.sqrt(dist_sq)**3
                
                # Magnitud de la fuerza
                f_mag = G * mass[j] * inv_dist3
                
                # Acumular aceleración
                acc_x += f_mag * dx
                acc_y += f_mag * dy
                acc_z += f_mag * dz
                
        # Escribir el resultado final en la memoria global de la GPU
        acc[i, 0] = acc_x
        acc[i, 1] = acc_y
        acc[i, 2] = acc_z

@cuda.jit
def update_kinematics_cuda(pos, vel, acc, dt):
    i = cuda.grid(1)
    if i < pos.shape[0]:
        # Método de Euler simpléctico
        vel[i, 0] += acc[i, 0] * dt
        vel[i, 1] += acc[i, 1] * dt
        vel[i, 2] += acc[i, 2] * dt
        
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt

# --- 2. CONFIGURACIÓN DEL HOST (CPU) Y DEVICE (GPU) ---

N_BODIES = 1000
G = 1.0
SOFTENING = 0.1
DT = 0.01

# Inicializar datos en la CPU usando NumPy
np.random.seed(42)
pos_host = np.random.randn(N_BODIES, 3).astype(np.float32) * 2.0
vel_host = np.random.randn(N_BODIES, 3).astype(np.float32) * 0.5
mass_host = (np.random.rand(N_BODIES).astype(np.float32) * 10.0 + 1.0)
acc_host = np.zeros((N_BODIES, 3), dtype=np.float32)

# Transferir arreglos de la CPU a la memoria de la GPU
pos_device = cuda.to_device(pos_host)
vel_device = cuda.to_device(vel_host)
mass_device = cuda.to_device(mass_host)
acc_device = cuda.to_device(acc_host)

# Configurar la cuadrícula de ejecución (Grid y Blocks) de CUDA
threads_per_block = 256
blocks_per_grid = (N_BODIES + (threads_per_block - 1)) // threads_per_block

# --- 3. VISUALIZACIÓN ---

fig = plt.figure(figsize=(8, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

scatter = ax.scatter(pos_host[:, 0], pos_host[:, 1], pos_host[:, 2], 
                     s=mass_host*2, c='cyan')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.axis('off')

def update(frame):
    # Ejecutar varios pasos de integración gráfica por frame
    for _ in range(5):
        # Lanzar Kernel de aceleración
        compute_accelerations_cuda[blocks_per_grid, threads_per_block](
            pos_device, mass_device, acc_device, SOFTENING, G
        )
        # Sincronizar para asegurar que la aceleración se calculó antes de mover
        cuda.synchronize() 
        
        # Lanzar Kernel de cinemática
        update_kinematics_cuda[blocks_per_grid, threads_per_block](
            pos_device, vel_device, acc_device, DT
        )
        cuda.synchronize()

    # Copiar de vuelta a la CPU SÓLO la posición para poder dibujarla
    pos_device.copy_to_host(pos_host)
    
    scatter._offsets3d = (pos_host[:, 0], pos_host[:, 1], pos_host[:, 2])
    return scatter,

ani = animation.FuncAnimation(fig, update, frames=300, interval=20, blit=False)
plt.show()