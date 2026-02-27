from sklearn.mixture import GaussianMixture

def train_gmm(X, n_components, covariance_type='full'):
    """Trains a Gaussian Mixture Model (GMM).

    Unlike K-means, GMM returns membership probabilities (soft clustering).
    
    Args:
        X (numpy.ndarray): Data matrix (n_samples, n_features).
        n_components (int): Number of Gaussians (clusters).
        covariance_type (str): The geometric shape of the clusters.
            - 'full': Each component has its own general covariance matrix (free-form ellipses).
            - 'tied': All components share the same covariance matrix.
            - 'diag': Ellipses are aligned with the axes (similar to Naive Bayes).
            - 'spherical': Each component has its own single variance (circular clusters).

    Returns:
        tuple: (probabilities, means)
            - probabilities (numpy.ndarray): Matrix (n_samples, n_components) representing the certainty of membership.
            - means (numpy.ndarray): Centers of the Gaussians (equivalent to centroids).
    """
    # Initialize with a fixed random_state for educational reproducibility
    gmm = # TODO
    
    gmm.fit(X)
    
    # Extract membership probabilities
    probabilities = gmm.predict_proba(X)
    
    return probabilities, gmm.means_