import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange

# --- 1. CONFIGURACIÓN FÍSICA Y PARÁMETROS ---
N_BODIES = 100      # Número de cuerpos
G = 1.0               # Constante gravitacional (simplificada)
SOFTENING = 0.1       # Parámetro de suavizado para evitar colisiones infinitas
DT = 0.01             # Paso de tiempo (Delta t)

# --- 2. NÚCLEO MATEMÁTICO ---
# Usamos njit para compilar en CPU ahora. 
# En el futuro, esto se convertirá en un @cuda.jit
@njit
def compute_accelerations(pos, mass, acc):
    # pos: (N, 3), mass: (N,), acc: (N, 3)
    n = pos.shape[0]
    
    # Reiniciar aceleraciones a cero
    for i in range(n):
        acc[i, 0] = 0.0
        acc[i, 1] = 0.0
        acc[i, 2] = 0.0
        
    # Calcular interacciones por pares
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
                
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            
            # Distancia al cuadrado + suavizado
            dist_sq = dx**2 + dy**2 + dz**2 + SOFTENING**2
            
            # 1 / distancia al cubo (optimizando la raíz cuadrada)
            inv_dist3 = (dist_sq)**(-1.5)
            
            # Fuerza = G * m_j / dist^3
            f_mag = G * mass[j] * inv_dist3
            
            acc[i, 0] += f_mag * dx
            acc[i, 1] += f_mag * dy
            acc[i, 2] += f_mag * dz

@njit
def symplectic_euler_step(pos, vel, acc, mass, dt):
    # 1. Calcular aceleraciones actuales
    compute_accelerations(pos, mass, acc)
    
    n = pos.shape[0]
    for i in prange(n):
        # 2. Actualizar velocidades (v = v + a*dt)
        vel[i, 0] += acc[i, 0] * dt
        vel[i, 1] += acc[i, 1] * dt
        vel[i, 2] += acc[i, 2] * dt
        
        # 3. Actualizar posiciones (r = r + v*dt)
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt

# --- 3. INICIALIZACIÓN DE DATOS ---
# Generamos posiciones iniciales aleatorias en una esfera y velocidades tangenciales
np.random.seed(42)
pos = np.random.randn(N_BODIES, 3) * 2.0
vel = np.random.randn(N_BODIES, 3) * 0
mass = np.random.rand(N_BODIES) * 10.0 + 1.0
acc = np.zeros((N_BODIES, 3))

# --- 3.5 LLAMADA EN FRÍO (WARM-UP) ---

# Pasamos los arrays reales, pero con dt=0.0 para que nada se mueva.
# Como compute_accelerations está dentro de esta función, Numba compilará AMBAS.
symplectic_euler_step(pos, vel, acc, mass, 0.0)

# --- 4. VISUALIZACIÓN DINÁMICA ---
fig = plt.figure(figsize=(8, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# Configurar el gráfico
scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                     s=mass*2, c='cyan', depthshade=True)

# Límites del mapa
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.axis('off') # Ocultar ejes para que parezca el espacio

def update(frame):
    # Ejecutamos varios pasos de física por cada frame de video 
    # para que la animación fluya a buena velocidad
    for _ in range(5):
        symplectic_euler_step(pos, vel, acc, mass, DT)
    
    # Actualizar la posición de los puntos en el gráfico
    scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    return scatter,

# Crear animación
ani = animation.FuncAnimation(fig, update, frames=200, interval=20, blit=False)
plt.show()