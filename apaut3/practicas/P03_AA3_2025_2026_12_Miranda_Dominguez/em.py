from scipy.stats import multivariate_normal
import numpy as np

def e_step(X, pi, mu, cov):
    """
    Paso E: calcula las responsabilidades r_nk = P(z_n=k | x_n, theta)
    usando el prior categórico pi_k y la verosimilitud gaussiana.
    """
    N, K = X.shape[0], len(pi)
    R = np.zeros((N, K))
    for k in range(K):
        R[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=cov[k])
    R /= R.sum(axis=1, keepdims=True)   # normalización → P(z_n=k | x_n)
    return R

def m_step(X, R, eps=0.0):
    """
    Paso M: actualiza theta usando las responsabilidades fijas del paso E.

    - pi_k  = N_k / N                              (fracción efectiva)
    - mu_k  = sum_n r_nk * x_n / N_k              (media ponderada)
    - Sigma_k = sum_n r_nk (x_n-mu_k)(x_n-mu_k)^T / N_k  (cov ponderada)

    Estas fórmulas son la solución cerrada que garantiza la familia exponencial.
    """
    N, D = X.shape
    K    = R.shape[1]
    N_k  = R.sum(axis=0)                           # (K,)  — puntos efectivos por componente

    pi_new  = N_k / N
    mu_new  = (R.T @ X) / N_k[:, None]             # (K, D)
    cov_new = np.zeros((K, D, D))
    for k in range(K):
        diff        = X - mu_new[k]                # (N, D)
        cov_new[k]  = (R[:, k, None] * diff).T @ diff / N_k[k]
        cov_new[k] += eps * np.eye(D)   


    return pi_new, mu_new, cov_new

def log_likelihood(X, pi, mu, cov):
    """log P(X|theta) = sum_n log sum_k pi_k * N(x_n | mu_k, Sigma_k)"""
    N  = X.shape[0]
    ll = np.zeros(N)
    for k in range(len(pi)):
        ll += pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=cov[k])
    return np.sum(np.log(ll + 1e-300))