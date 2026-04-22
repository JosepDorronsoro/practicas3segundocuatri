import torch
import numpy as np
import score_model as sm


SIGMA_BM = 25.0

def bm_drift_coefficient(x_t, t):
    """Drift nulo para movimiento browniano."""
    return torch.zeros_like(x_t)

def bm_diffusion_coefficient(t, sigma=SIGMA_BM):
    """g(t) = sigma^t"""
    return sigma**t

def bm_mu_t(x_0, t):
    """Media de la perturbación: p(x_t|x_0) = N(x_t; x_0, sigma_t^2)"""
    return x_0

def bm_sigma_t(t, sigma=SIGMA_BM):
    """Desviación típica de la perturbación acumulada."""
    # t puede ser un escalar o un tensor expandido [batch, 1, 1, 1]
    return torch.sqrt(0.5 * (sigma**(2 * t) - 1.0) / np.log(sigma))

# 2. Función para inicializar el modelo de score que subiste
def crear_score_model(device):
    # El modelo UNet de tu archivo score_model.py
    # marginal_prob_std es la función sigma_t
    model = sm.ScoreNet(marginal_prob_std=bm_sigma_t).to(device)
    return model