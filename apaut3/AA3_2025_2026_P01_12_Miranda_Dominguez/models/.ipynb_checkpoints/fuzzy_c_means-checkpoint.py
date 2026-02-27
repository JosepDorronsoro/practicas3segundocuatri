import numpy as np
import warnings
from scipy.spatial.distance import cdist


def train_fuzzy_c_means(X, n_clusters, m=2.0, max_iters=100, epsilon=1e-5):
    """Implements Fuzzy C-Means in a compact and vectorized way.

    This function uses matrix operations to optimize membership
    and centroids in a single logical block.

    Args:
        X (numpy.ndarray): Training data (n_samples, n_features).
        n_clusters (int): Number of clusters.
        m (float): Fuzziness parameter.
        max_iters (int): Maximum iterations.
        epsilon (float): Convergence tolerance.

    Returns:
        tuple: (U, centroids) where U is the membership matrix 
        and centroids is the matrix of cluster centers.
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