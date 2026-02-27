import numpy as np
import warnings
from scipy.spatial.distance import cdist
from typing import Tuple

def train_fuzzy_c_means(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iters: int = 100,
    epsilon: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements Fuzzy C-Means in a compact and vectorized way.

    Args:
        X: Training data of shape (n_samples, n_features).
        n_clusters: Number of clusters.
        m: Fuzziness parameter.
        max_iters: Maximum number of iterations.
        epsilon: Convergence tolerance.

    Returns:
        A tuple (U, centroids) where:
            U is the membership matrix of shape (n_samples, n_clusters),
            centroids is the matrix of cluster centers of shape (n_clusters, n_features).
    """

    # 1. Initialization: Random normalized membership matrix
    n_samples = X.shape[0]
    W = np.random.rand(n_samples, n_clusters)
    W /= W.sum(axis=1)[:, np.newaxis]

    n_iter = 0
    stop_criterion = False
    
    while n_iter < max_iters and not stop_criterion:
        

        W_old = W.copy()
        
        # 2. Optimization Step: Calculate Centroids broadcasted
        # cj = sum(w_ij^m * xi) / sum(w_ij^m)
       
        
        # 3. Optimization Step: Update Membership (U)
        # Calculate distances from all points to all centroids using cdist
        
        
        # Vectorized formula: w_ij = 1 / sum((d_ij/d_ik)^(2/(m-1))) 
   
        
        # 4. Stopping criterion
        
        n_iter += 1
            
    
    if n_iter == max_iters: 
        warnings.warn("Maximum number of iterations reached.", RuntimeWarning)
            
    return U, centroids