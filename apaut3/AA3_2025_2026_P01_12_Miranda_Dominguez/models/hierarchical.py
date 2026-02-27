# Install if necessary: !pip install ucimlrepo
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

def train_hierarchical(X):
    """
    Implements Hierarchical Clustering using Ward's method.

    This method minimizes the variance within clusters, making it 
    suitable for data with numerical and scaled features.

    Args:
        X (numpy.ndarray): Training data (n_samples, n_features).

    Returns:
        Z (numpy.ndarray): The linkage matrix resulting from hierarchical clustering.
    """
    # Scale data to improve clustering quality
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Ward's method for hierarchical clustering
    # Ward minimizes the sum of squared differences within all clusters
    Z = # TODO: Use scipy's linkage function with method='ward' to compute the hierarchical clustering
    
    return Z

def plot_hierarchical(animal_names, Z):
    """
    Visualizes the dendrogram of the hierarchical clustering.

    Args:
        animal_names (list): List of animal names to label the leaves.
        Z (numpy.ndarray): The linkage matrix resulting from hierarchical clustering.
    """
    plt.figure(figsize=(18, 10))
    
    # Create the Dendrogram: 
    # TODO: Use scipy's dendrogram function to plot the hierarchical clustering
    
    plt.title("Animal Taxonomy: Hierarchical Clustering (Source: UCI Machine Learning)", fontsize=16)
    plt.ylabel("Dissimilarity Distance (Ward)", fontsize=12)
    plt.xlabel("Animal Species", fontsize=12)
    
    # Visual guide: Suggested cut-off line to define clusters
    plt.axhline(y=17, color='r', linestyle='--') 
    
    plt.grid(axis='y', alpha=0.3)
    plt.show()