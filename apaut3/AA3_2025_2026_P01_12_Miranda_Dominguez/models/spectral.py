from sklearn.cluster import SpectralClustering

def train_spectral(X, n_clusters, affinity='nearest_neighbors', n_neighbors=10):
    """Train Spectral Clustering based on graphs.

    Uses eigenvalues of the affinity matrix to reduce dimensionality 
    before applying clustering. Ideal for connected but non-convex clusters 
    (rings, elongated shapes).

    Args:
        X (numpy.ndarray): Input data.
        n_clusters (int): Number of clusters to find.
        affinity (str): How to construct the graph. 
                        'nearest_neighbors' works well for complex geometric shapes.
                        'rbf' uses Gaussian kernel.
        n_neighbors (int): Number of neighbors for graph construction.

    Returns:
        numpy.ndarray: Cluster labels.
    """
    
    model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors)
    labels = model.fit_predict(X)
    return labels
