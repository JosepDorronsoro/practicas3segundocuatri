# Install if necessary: !pip install ucimlrepo
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
import numpy as np
import typing

def train_hierarchical(X: np.ndarray) -> np.ndarray:
    """
    Implements Hierarchical Clustering using Ward's method.

    This method minimizes the variance within clusters, making it 
    suitable for data with numerical and scaled features.

    Args:
        X (numpy.ndarray): Training data (n_samples, n_features).

    Returns:
        Z (numpy.ndarray): The linkage matrix resulting from hierarchical clustering.
    """

    # Apply Ward's method for hierarchical clustering
    # Ward minimizes the sum of squared differences within all clusters
    Z = linkage(X, method='ward')
    
    return Z

def plot_hierarchical(animal_names: list, Z: np.ndarray):

    """
    Visualizes the dendrogram of the hierarchical clustering.

    Args:
        animal_names (list): List of animal names to label the leaves.
        Z (numpy.ndarray): The linkage matrix resulting from hierarchical clustering.
        
    """
    plt.figure(figsize=(30, 15))
    
    # Create the Dendrogram: 
    dendrogram(Z, labels=animal_names, leaf_font_size=16)

    plt.title("Animal Taxonomy: Hierarchical Clustering (Source: UCI Machine Learning)", fontsize=16)
    plt.ylabel("Dissimilarity Distance (Ward)", fontsize=12)
    plt.xlabel("Animal Species", fontsize=12)
    
    # Visual guide: Suggested cut-off line to define clusters
    plt.axhline(y=17, color='r', linestyle='--') 
    plt.grid(axis='y', alpha=0.3)
    plt.show()