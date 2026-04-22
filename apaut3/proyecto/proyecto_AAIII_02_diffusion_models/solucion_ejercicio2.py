import torch
import numpy as np
from functools import partial
import diffusion_process as dfp

# Parámetros por defecto
SIGMA_DEFAULT = 25.0

def bm_drift_coefficient(x_t, t):
    """Coeficiente de deriva para el movimiento browniano (f=0)."""
    return torch.zeros_like(x_t)

def bm_diffusion_coefficient(t, sigma=SIGMA_DEFAULT):
    """Coeficiente de difusión g(t) = sigma^t."""
    return sigma**t

def bm_mu_t(x_0, t):
    """Media de la perturbación p(x_t|x_0)."""
    return x_0

def bm_sigma_t(t, sigma=SIGMA_DEFAULT):
    """Desviación típica de la perturbación acumulada."""
    return torch.sqrt(0.5 * (sigma**(2 * t) - 1.0) / np.log(sigma))

def backward_drift_coefficient(x_t, t, score_model, diffusion_coefficient):
    """
    Calcula el drift reverso para el integrador: f_rev = -(g_t^2) * score.
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], device=x_t.device, dtype=torch.float32)
    
    if t.dim() == 0 or t.shape[0] != x_t.shape[0]:
        t = t.expand(x_t.shape[0])
    
    score = score_model(x_t, t)
    
    # Ajustar dimensiones de g_t para operar con el batch de imágenes
    view_shape = [x_t.shape[0]] + [1] * (x_t.dim() - 1)
    g_t = diffusion_coefficient(t).view(*view_shape)
    
    return -(g_t ** 2) * score

def generar_muestras_ejercicio(score_model, n_images, device='cpu'):
    """
    Encapsula la lógica de generación completa usando Euler-Maruyama.
    """
    # 1. Configurar el proceso de difusión
    proceso = dfp.GaussianDiffussionProcess(
        drift_coefficient=bm_drift_coefficient,
        diffusion_coefficient=bm_diffusion_coefficient,
        mu_t=bm_mu_t,
        sigma_t=bm_sigma_t,
    )

    T = 1.0
    t_tensor = torch.tensor([T], device=device)
    
    # 2. Preparar el ruido inicial (Prior)
    sigma_max = proceso.sigma_t(t_tensor).detach()
    # Nota: Ajustado a (1, 28, 28) para MNIST estándar, 
    # cambia a (3, 32, 32) si usas el modelo de color.
    channels, h, w = (1, 28, 28) if score_model.conv1.in_channels == 1 else (3, 32, 32)
    image_T = torch.randn(n_images, channels, h, w, device=device) * sigma_max.view(1, 1, 1, 1)

    # 3. Integración Euler-Maruyama
    with torch.no_grad():
        times, synthetic_images_t = dfp.euler_maruyama_integrator(
            image_T,
            t_0=T,
            t_end=1.0e-3,
            n_steps=500,
            drift_coefficient=partial(
                backward_drift_coefficient,
                score_model=score_model,
                diffusion_coefficient=bm_diffusion_coefficient,
            ),
            diffusion_coefficient=bm_diffusion_coefficient,
        )
    
    return synthetic_images_t