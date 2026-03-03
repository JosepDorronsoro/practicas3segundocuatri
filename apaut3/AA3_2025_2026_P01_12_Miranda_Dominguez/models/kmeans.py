import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def evaluate_kmeans_hyperparameters(X, k_range):
    """
    Calculates Inertia and Silhouette metrics for a range of k values.

    Args:
        X (numpy.ndarray): Feature matrix.
        k_range (list): List of k values (number of clusters) to evaluate.

    Returns:
        tuple: (inertias, silhouette_scores) containing the metrics for each k.
    """
    inertias = []
    silhouette_scores = []

    for k in k_range:
        # We use random initialization and a fixed seed for educational consistency
        kmeans = KMeans(n_clusters=k, init='random', random_state=42).fit(X)
        inertias.append(kmeans.inertia_)
        
        # Silhouette score requires at least 2 clusters to be calculated
        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(None)
        
    return inertias, silhouette_scores

def train_kmeans(X, n_clusters):
    """
    Trains a K-means model and returns predicted labels and centroids.

    This function encapsulates initialization, fitting, and label extraction.

    Args:
        X (numpy.ndarray): Feature matrix (n_samples, n_features).
        n_clusters (int): Number of clusters to find.

    Returns:
        tuple: (labels, centroids) where:
            - labels (numpy.ndarray): Vector of labels assigned to each point.
            - centroids (numpy.ndarray): Coordinates of the center of each cluster.
    """

    model = KMeans(n_clusters=n_clusters)
    
    # Model fitting
    model.fit(X)
    
    return model.labels_, model.cluster_centers_