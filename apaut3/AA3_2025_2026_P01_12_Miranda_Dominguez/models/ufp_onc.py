import numpy as np
import warnings
from scipy.spatial.distance import cdist
from typing import Tuple, Optional


# ─────────────────────────────────────────────
#  Layer 1 – Fuzzy K-Means  (Euclidean distance)
# ─────────────────────────────────────────────

def _fuzzy_kmeans(
    X: np.ndarray,
    centroids: np.ndarray,
    m: float = 2.0,
    max_iters: int = 100,
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """One run of fuzzy K-means starting from given centroids (Euclidean distance).

    Args:
        X: Data matrix of shape (n_samples, n_features).
        centroids: Initial cluster centers of shape (n_clusters, n_features).
        m: Fuzziness exponent (q in the paper).
        max_iters: Maximum number of iterations.
        epsilon: Convergence tolerance on the membership matrix.

    Returns:
        A tuple (U, centroids) where:
            U         – membership matrix of shape (n_samples, n_clusters),
            centroids – updated cluster centers of shape (n_clusters, n_features).
    """
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]

    # Bootstrap: build initial U from provided centroids using Eq.(2)
    d = cdist(X, centroids, metric='euclidean') + 1e-10
    U = 1.0 / np.sum(
        (d[:, :, np.newaxis] / d[:, np.newaxis, :]) ** (2.0 / (m - 1)),
        axis=2,
    )

    n_iter = 0
    stop_criterion = False

    while n_iter < max_iters and not stop_criterion:
        U_old = U.copy()

        # Eq.(3) – update centroids
        U_m = np.power(U_old, m)
        centroids = U_m.T @ X / U_m.sum(axis=0)[:, np.newaxis]

        # Eq.(2) – update memberships
        d = cdist(X, centroids, metric='euclidean') + 1e-10
        U = 1.0 / np.sum(
            (d[:, :, np.newaxis] / d[:, np.newaxis, :]) ** (2.0 / (m - 1)),
            axis=2,
        )

        # Eq.(4) – stopping criterion
        stop_criterion = np.max(np.abs(U_old - U)) < epsilon
        n_iter += 1

    if n_iter == max_iters:
        warnings.warn("Fuzzy K-Means: maximum iterations reached.", RuntimeWarning)

    # Final centroid update
    U_m = np.power(U, m)
    centroids = U_m.T @ X / U_m.sum(axis=0)[:, np.newaxis]

    return U, centroids


# ─────────────────────────────────────────────
#  Layer 2 – FMLE  (exponential / Mahalanobis distance)
# ─────────────────────────────────────────────

def _fuzzy_covariance(
    X: np.ndarray,
    H: np.ndarray,
    centroids: np.ndarray,
) -> list[np.ndarray]:
    """Eq.(9) – fuzzy covariance matrices for each cluster.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        H: Posterior probability matrix h(i|Xj) of shape (n_samples, n_clusters).
        centroids: Cluster centers of shape (n_clusters, n_features).

    Returns:
        List of covariance matrices, one per cluster, each (n_features, n_features).
    """
    n_clusters = centroids.shape[0]
    n_features = X.shape[1]
    covariances = []

    for i in range(n_clusters):
        h_i = H[:, i]                            # (n_samples,)
        diff = X - centroids[i]                  # (n_samples, n_features)
        # weighted outer-product sum divided by sum of weights – Eq.(9)
        F_i = (h_i[:, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0)
        F_i /= h_i.sum() + 1e-10
        # Regularise to keep matrix positive-definite.
        # Scale regularisation to the matrix trace so it stays meaningful
        # regardless of the data scale – this prevents singular covariances
        # when a cluster collapses to very few points.
        F_i += np.eye(n_features) * (np.trace(F_i) * 1e-4 + 1e-6)
        covariances.append(F_i)

    return covariances


def _exponential_distance(
    X: np.ndarray,
    centroids: np.ndarray,
    covariances: list[np.ndarray],
    priors: np.ndarray,
) -> np.ndarray:
    """Eq.(7) – exponential distance d²_e(Xj, Vi) for each (sample, cluster) pair.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        centroids: Cluster centers of shape (n_clusters, n_features).
        covariances: List of fuzzy covariance matrices, one per cluster.
        priors: A-priori cluster probabilities of shape (n_clusters,).

    Returns:
        Distance matrix of shape (n_samples, n_clusters).
    """
    n_samples  = X.shape[0]
    n_clusters = centroids.shape[0]
    D = np.zeros((n_samples, n_clusters))

    for i in range(n_clusters):
        F_i   = covariances[i]
        F_inv = np.linalg.inv(F_i)
        diff  = X - centroids[i]                        # (n_samples, n_features)
        # Mahalanobis term: (Xj - Vi)^T F_i^{-1} (Xj - Vi) – Eq.(7)
        maha  = np.einsum('ni,ij,nj->n', diff, F_inv, diff)
        # Clip maha before exp() to prevent overflow (large maha → very large
        # distance, which is fine numerically once capped at exp(500))
        maha  = np.clip(maha, 0.0, 1000.0)
        det   = max(np.linalg.det(F_i), 1e-300)
        # d²_e = sqrt(det(Fi)) / Pi * exp(maha / 2) – Eq.(7)
        D[:, i] = (det ** 0.5) / (priors[i] + 1e-10) * np.exp(maha / 2.0)

    return D


def _fmle(
    X: np.ndarray,
    centroids: np.ndarray,
    max_iters: int = 100,
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Fuzzy Maximum Likelihood Estimation (FMLE) – Eqs.(6-9).

    Args:
        X: Data matrix of shape (n_samples, n_features).
        centroids: Initial cluster centers of shape (n_clusters, n_features).
        max_iters: Maximum number of iterations.
        epsilon: Convergence tolerance on the posterior probability matrix.

    Returns:
        A tuple (H, centroids, covariances, priors) where:
            H           – posterior probability matrix of shape (n_samples, n_clusters),
            centroids   – updated cluster centers of shape (n_clusters, n_features),
            covariances – list of fuzzy covariance matrices (one per cluster),
            priors      – a-priori cluster probabilities of shape (n_clusters,).
    """
    n_samples, n_features = X.shape
    n_clusters = centroids.shape[0]

    # Uniform priors and identity covariances as starting point
    priors      = np.full(n_clusters, 1.0 / n_clusters)
    covariances = [np.eye(n_features) for _ in range(n_clusters)]

    # Bootstrap H with Euclidean distances before iterating
    d_init = cdist(X, centroids, metric='euclidean') + 1e-10
    H = 1.0 / np.sum(
        (d_init[:, :, np.newaxis] / d_init[:, np.newaxis, :]) ** 2.0,
        axis=2,
    )

    n_iter = 0
    stop_criterion = False

    while n_iter < max_iters and not stop_criterion:
        H_old = H.copy()

        # Eq.(3) – update centroids (same weighted-mean formula as K-means)
        H_denom = H_old.sum(axis=0)[:, np.newaxis] + 1e-10
        centroids = (H_old.T @ X) / H_denom

        # Eq.(9) – update fuzzy covariance matrices
        covariances = _fuzzy_covariance(X, H_old, centroids)

        # Eq.(8) – update a-priori probabilities
        priors = H_old.mean(axis=0)

        # Eq.(7) – exponential distances, Eq.(6) – new posteriors
        D = _exponential_distance(X, centroids, covariances, priors)
        H = 1.0 / np.sum(
            (D[:, :, np.newaxis] / (D[:, np.newaxis, :] + 1e-10)),
            axis=2,
        )

        stop_criterion = np.max(np.abs(H_old - H)) < epsilon
        n_iter += 1

    if n_iter == max_iters:
        warnings.warn("FMLE: maximum iterations reached.", RuntimeWarning)

    return H, centroids, covariances, priors


# ─────────────────────────────────────────────
#  Performance measures – Section II-C
# ─────────────────────────────────────────────

def _fuzzy_hypervolume(covariances: list[np.ndarray]) -> float:
    """Eq.(10) – Fuzzy HyperVolume (FHV): sum of sqrt(det(Fi)).

    Args:
        covariances: List of fuzzy covariance matrices.

    Returns:
        Scalar FHV value.
    """
    return sum(max(np.linalg.det(F), 0.0) ** 0.5 for F in covariances)


def _partition_density(
    X: np.ndarray,
    H: np.ndarray,
    centroids: np.ndarray,
    covariances: list[np.ndarray],
) -> Tuple[float, float]:
    """Eqs.(11-14) – Average Partition Density (D_PA) and Partition Density (PD).

    Only data points inside the unit hyperellipsoid of each cluster
    (Mahalanobis distance < 1) are considered 'central members'.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        H: Posterior probability matrix of shape (n_samples, n_clusters).
        centroids: Cluster centers of shape (n_clusters, n_features).
        covariances: List of fuzzy covariance matrices.

    Returns:
        A tuple (D_PA, PD) – average and global partition density.
    """
    n_clusters = centroids.shape[0]
    fhv = _fuzzy_hypervolume(covariances)

    S_total = 0.0
    D_PA_terms = []

    for i in range(n_clusters):
        F_inv = np.linalg.inv(covariances[i])
        diff  = X - centroids[i]
        maha  = np.einsum('ni,ij,nj->n', diff, F_inv, diff)  # Eq.(14) condition

        # Eq.(12) – S_i: sum of memberships for central members only
        inside = maha < 1.0
        S_i = H[inside, i].sum()
        S_total += S_i

        # Eq.(11) – D_PA contribution from cluster i
        det_i = max(np.linalg.det(covariances[i]), 1e-300)
        D_PA_terms.append(S_i / (det_i ** 0.5 + 1e-10))

    D_PA = np.mean(D_PA_terms)              # Eq.(11)
    PD   = S_total / (fhv + 1e-10)         # Eq.(13)

    return D_PA, PD


# ─────────────────────────────────────────────
#  Unsupervised prototype tracking – Section II-B
# ─────────────────────────────────────────────

def _init_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """Unsupervised centroid initialisation following Section II-B.

    Places the first prototype at the data mean and grows the set by
    iteratively inserting a 'non-physical' point far from all existing
    members, then running one pass of fuzzy K-means to let centroids
    settle before the next prototype is added.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        k: Target number of cluster prototypes.

    Returns:
        Initial centroids of shape (k, n_features).
    """
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-10

    # Step 2 – first prototype at the global mean
    centroids = mean[np.newaxis, :]

    for _ in range(1, k):
        # Step 3 – nonphysical point at (mean + 3*std) from current centroids
        nonphysical = mean + 3.0 * std
        centroids   = np.vstack([centroids, nonphysical])

        # Step 4 – one quick fuzzy K-means pass to settle centroids
        _, centroids = _fuzzy_kmeans(X, centroids, max_iters=20, epsilon=1e-3)

    return centroids


# ─────────────────────────────────────────────
#  UFP-ONC – main entry point
# ─────────────────────────────────────────────

def train_ufp_onc(
    X: np.ndarray,
    k_max: int = 8,
    m: float = 2.0,
    max_iters: int = 100,
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    """UFP-ONC: Unsupervised Fuzzy Partition – Optimal Number of Classes.

    Combines fuzzy K-means (layer 1) and FMLE (layer 2) with unsupervised
    prototype tracking. Iterates over k = 2 … k_max clusters and selects
    the optimal k via the FHV (minimum) and PD (maximum) performance measures.

    Args:
        X: Training data of shape (n_samples, n_features).
        k_max: Maximum number of clusters to evaluate.
        m: Fuzziness exponent (q in the paper, fixed at 2 as per Section III).
        max_iters: Maximum iterations for inner optimisation loops.
        epsilon: Convergence tolerance.

    Returns:
        A tuple (U_opt, centroids_opt, k_opt, history) where:
            U_opt        – membership matrix for optimal k, shape (n_samples, k_opt),
            centroids_opt – cluster centers for optimal k, shape (k_opt, n_features),
            k_opt        – estimated optimal number of clusters,
            history      – dict with keys 'fhv', 'D_PA', 'PD' listing the metric
                           value at each k from 2 to k_max.
    """
    history = {'k': [], 'fhv': [], 'D_PA': [], 'PD': []}
    results = {}

    for k in range(2, k_max + 1):

        # ── Step 1: fuzzy K-means with unsupervised prototype tracking
        init_c = _init_centroids(X, k)
        U_km, centroids_km = _fuzzy_kmeans(X, init_c, m=m,
                                           max_iters=max_iters, epsilon=epsilon)

        # ── Step 2: FMLE refinement starting from K-means centroids
        H, centroids, covariances, priors = _fmle(X, centroids_km,
                                                   max_iters=max_iters,
                                                   epsilon=epsilon)

        # ── Step 3: performance measures
        fhv        = _fuzzy_hypervolume(covariances)
        D_PA, PD   = _partition_density(X, H, centroids, covariances)

        history['k'].append(k)
        history['fhv'].append(fhv)
        history['D_PA'].append(D_PA)
        history['PD'].append(PD)

        results[k] = (H, centroids, covariances, priors)

    # ── Step 4: optimal k – joint criterion: normalised FHV (↓) + normalised PD (↑)
    # Using only FHV is brittle when the curve is noisy; combining both measures
    # (as the paper intends) gives a much more robust estimate of k_opt.
    fhv_arr = np.array(history['fhv'],  dtype=float)
    PD_arr  = np.array(history['PD'],   dtype=float)

    # Normalise each metric to [0, 1] so they are comparable
    fhv_norm = (fhv_arr - fhv_arr.min()) / (fhv_arr.ptp() + 1e-10)  # lower is better
    PD_norm  = (PD_arr  - PD_arr.min())  / (PD_arr.ptp()  + 1e-10)  # higher is better

    # Combined score: minimise FHV, maximise PD → minimise (fhv_norm - PD_norm)
    combined = fhv_norm - PD_norm
    k_opt    = history['k'][int(np.argmin(combined))]

    H_opt, centroids_opt, _, _ = results[k_opt]

    return H_opt, centroids_opt, k_opt, history