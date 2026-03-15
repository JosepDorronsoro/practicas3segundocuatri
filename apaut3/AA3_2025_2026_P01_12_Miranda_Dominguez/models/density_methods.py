from sklearn.cluster import DBSCAN, HDBSCAN
import numpy as np

def train_dbscan(X, eps=0.5, min_samples=5):
    """Trains DBSCAN to find clusters of arbitrary shapes.

    Args:
        X (numpy.ndarray): Input data.
        eps (float): The neighborhood radius (maximum distance between two samples).
        min_samples (int): The number of samples in a neighborhood for a point 
                           to be considered a core point.

    Returns:
        numpy.ndarray: Cluster labels. Noise points are labeled as -1.
    """
    
    # DBSCAN is deterministic, so it does not require a random_state
    db = DBSCAN(eps=eps, min_samples=min_samples) 
    labels = db.fit_predict(X)
    return labels

def train_hdbscan(X, min_cluster_size=5):
    
    """Trains HDBSCAN (Hierarchical DBSCAN).

    Ideal for datasets with variable densities. It does not require a fixed eps.

    Args:
        X (numpy.ndarray): Input data.
        min_cluster_size (int): The minimum number of samples in a group 
                                to be considered a cluster.

    Returns:
        numpy.ndarray: Cluster labels. Noise points are labeled as -1.
    """
    # Note: Requires scikit-learn >= 1.3.0
    hdb = HDBSCAN(min_cluster_size)
    labels = hdb.fit(X)
    return labels.labels_

def calculate_wcss(X, labels):
    """Manually calculates WCSS, ignoring noise points (-1)"""
    wcss = 0
    # Find all unique clusters, ignoring noise (-1)
    clusters = [label for label in np.unique(labels) if label != -1]
    
    for cluster in clusters:
        # Get all points in this cluster
        cluster_points = X[labels == cluster]
        # Find the center (centroid) of this specific cluster
        centroid = np.mean(cluster_points, axis=0)
        # Add the squared distances from points to the centroid
        wcss += np.sum((cluster_points - centroid) ** 2)
        
    return wcss