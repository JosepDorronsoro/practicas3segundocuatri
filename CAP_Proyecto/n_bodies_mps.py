import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. CONFIGURACIÓN DEL DISPOSITIVO ---
# Detectar si MPS (Metal Performance Shaders) está disponible
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("✓ Ejecutando en GPU de Apple usando MPS (Metal)")
else:
    device = torch.device("cpu")
    print("✗ MPS no disponible. Usando CPU (será lento)")

# --- 2. PARÁMETROS FÍSICOS ---
N_BODIES = 4       # ¡Ahora podemos manejar muchos más cuerpos gracias a la GPU!
G = 0.01               # Constante gravitacional
SOFTENING = 0.05      # Suavizado para evitar colisiones infinitas
DT = 0.01             # Paso de tiempo
DTYPE = torch.float32 # Usamos float32, que es lo nativo y más rápido en MPS

# --- 3. INICIALIZACIÓN DE DATOS DIRECTAMENTE EN EL DISPOSITIVO (MPS) ---
torch.manual_seed(42)

# Crear datos iniciales en CPU (NumPy es más fácil para inicializar formas)
pos_np = np.random.randn(N_BODIES, 3).astype(np.float32) * 2.0
vel_np = np.random.randn(N_BODIES, 3).astype(np.float32) * 0
# vel_np = np.zeros((3, 3)).astype(np.float32)
mass_np = (np.random.rand(N_BODIES).astype(np.float32) * 10.0 + 1.0) # Masas 1-11

# Mover datos a la GPU (MPS)
# Las posiciones y velocidades deben tener "requires_grad=False" ya que no estamos entrenando una IA.
pos = torch.from_numpy(pos_np).to(device)
vel = torch.from_numpy(vel_np).to(device)
mass = torch.from_numpy(mass_np).to(device)

# Pre-asignar tensor de aceleración en MPS para no crear memoria nueva en cada paso
acc = torch.zeros((N_BODIES, 3), device=device, dtype=DTYPE)

# --- 4. NÚCLEO FÍSICO VECTORIZADO (PyTorch/MPS) ---
def compute_accelerations_mps(pos, mass, softening):
    """Calcula aceleraciones de forma matricial (vectorizada) en MPS"""
    N = pos.shape[0]
    
    # Truco de Broadcasting para evitar bucles for:
    # pos_i shape: (N, 1, 3) - Cada fila 'i' es una posición
    # pos_j shape: (1, N, 3) - Cada columna 'j' es una posición
    pos_i = pos.view(N, 1, 3)
    pos_j = pos.view(1, N, 3)
    
    # dx, dy, dz shape: (N, N)
    # Contiene la diferencia de posición entre el cuerpo 'j' y el cuerpo 'i'
    # delta_pos shape es (N, N, 3)
    delta_pos = pos_j - pos_i 
    
    # Calcular distancias al cuadrado: (N, N)
    # Sumamos a lo largo del último eje (x,y,z)
    dist_sq = torch.sum(delta_pos**2, dim=2)
    
    # Añadir suavizado y calcular 1 / dist^3
    # dist_sq_softened shape: (N, N)
    dist_sq_softened = dist_sq + softening**2
    inv_dist3 = torch.rsqrt(dist_sq_softened)**3 # rsqrt es rápido: 1/sqrt(x)
    
    # Calcular la fuerza neta escalada. 
    # mass_j shape es (N). Al multiplicar por inv_dist3 (N,N), se hace broadcasting.
    # El resultado 'forces_scaled' es (N, N)
    forces_scaled = G * mass * inv_dist3
    
    # Aplicar la dirección de la fuerza.
    # forces_scaled (N,N) * delta_pos (N,N,3) -> (N,N,3)
    acc_by_pair = forces_scaled.unsqueeze(2) * delta_pos
    
    # Sumar las aceleraciones que todos los cuerpos 'j' ejercen sobre cada cuerpo 'i'
    # Sumamos a lo largo del eje 'j' (dim=1)
    # Resultado shape: (N, 3)
    acc_total = torch.sum(acc_by_pair, dim=1)
    
    return acc_total

def symplectic_euler_step_mps(pos, vel, mass, dt, softening):
    """Paso de integración física en MPS"""
    
    # 1. Calcular aceleraciones actuales en GPU
    acc = compute_accelerations_mps(pos, mass, softening)
    
    # 2. Actualizar velocidades in-place (v = v + a*dt)
    # Usar += o .add_ es crucial para no crear nuevos tensores y ahorrar memoria
    vel.add_(acc * dt)
    
    # 3. Actualizar posiciones in-place (r = r + v*dt)
    pos.add_(vel * dt)

# --- 5. VISUALIZACIÓN DINÁMICA (Matplotlib en CPU) ---
# OJO: Matplotlib no corre en MPS. Tenemos que bajar los datos de la GPU
# a la CPU para cada frame, lo cual es un cuello de botella, pero necesario
# para usar Matplotlib.

fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# Mover datos a CPU solo para la configuración inicial del scatter
pos_cpu = pos.cpu().numpy()
mass_cpu = mass.cpu().numpy()

scatter = ax.scatter(pos_cpu[:, 0], pos_cpu[:, 1], pos_cpu[:, 2], 
                     s=mass_cpu*3, c='cyan', edgecolors='white', linewidths=0.2, alpha=0.8)

# Límites del mapa (puedes ajustarlos o hacerlos dinámicos)
LIMIT = 6
ax.set_xlim(-LIMIT, LIMIT)
ax.set_ylim(-LIMIT, LIMIT)
ax.set_zlim(-LIMIT, LIMIT)
ax.axis('off')

def update(frame):
    # Ejecutamos varios pasos de física por frame en MPS para velocidad
    STEPS_PER_FRAME = 2
    for _ in range(STEPS_PER_FRAME):
        symplectic_euler_step_mps(pos, vel, mass, DT, SOFTENING)
    
    # --- CUELLO DE BOTELLA ---
    # Transferir datos de la GPU (MPS) a la CPU (NumPy) para dibujar
    pos_draw = pos.cpu().numpy()
    
    scatter._offsets3d = (pos_draw[:, 0], pos_draw[:, 1], pos_draw[:, 2])
    return scatter,

# Crear animación
# Usamos blit=False porque scatter 3D no soporta blitting eficiente en Matplotlib
ani = animation.FuncAnimation(fig, update, frames=500, interval=10, blit=False)

plt.show()