import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def visualize_clustering_comparison(X, y_true, y_pred, centroids=None):
    """
    Displays a visual comparison and highlights classification errors using label alignment.

    Automatically detects if y_pred consists of probabilities (soft clustering) or hard labels.
    Performs optimal label alignment to identify actual errors, regardless of the 
    cluster IDs assigned by the algorithm.

    Args:
        X (numpy.ndarray): Data features (n_samples, 2).
        y_true (numpy.ndarray): Ground Truth labels.
        y_pred (numpy.ndarray): Predicted labels or probability matrix.
        centroids (numpy.ndarray, optional): Cluster center coordinates.
    """
    # 1. Process y_pred (if it comes as probabilities from GMM/Fuzzy clustering)
    if y_pred.ndim == 2:
        # Take the most probable class for the base color
        y_pred_labels = np.argmax(y_pred, axis=1)
        # Probabilities could be used for transparency (alpha), here kept solid for clarity
        alphas = np.max(y_pred, axis=1)  
    else:
        y_pred_labels = y_pred.copy()
        alphas = np.ones(len(y_pred_labels))

    # 2. LABEL ALIGNMENT (The "Magic" Step)
    # K-means might call "Cluster 0" what is actually "Class 2". This step fixes that.
    cm = confusion_matrix(y_true, y_pred_labels)
    
    # Solve the linear sum assignment problem to maximize the match between labels
    rows, cols = linear_sum_assignment(cm, maximize=True) 
    
    # Create a translation map: {predicted_label: corresponding_real_label}
    label_mapping = {col: row for row, col in zip(rows, cols)}
    
    # Translate predictions into the "language" of the ground truth
    y_pred_aligned = np.array([label_mapping[val] for val in y_pred_labels])

    # 3. Error Detection
    # Now we can compare directly after alignment
    errors = y_pred_aligned != y_true
    error_indices = np.where(errors)[0]

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Ground Truth
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=40, alpha=0.7)
    ax1.set_title("Real Classification (Ground Truth)", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('True Class')

    # Plot 2: Prediction
    # A) Plot all points (colored according to aligned predictions)
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y_pred_aligned, cmap='viridis', s=40, alpha=0.7, label='Correct Prediction')

    # B) Overlay errors with a red border and transparent center
    if len(error_indices) > 0:
        ax2.scatter(X[error_indices, 0], X[error_indices, 1], 
                    facecolors='none',  # Transparent fill
                    edgecolors='red',   # Red border
                    linewidths=1.5,     # Thicker border
                    s=55,               # Slightly larger to encircle the point
                    label='Misclassified')

    # C) Centroids
    if centroids is not None:
        ax2.scatter(centroids[:, 0], centroids[:, 1], 
                    s=200, c='black', marker='X', edgecolors='white', linewidths=2, label='Centroids')

    ax2.set_title("Prediction (Red Border = Error)", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # --- CUSTOM LEGEND ---
    legend_elements = [
        mlines.Line2D([], [], color='white', marker='o', markerfacecolor='grey', markersize=10, label='Match'),
        mlines.Line2D([], [], color='white', marker='o', markeredgecolor='red', markerfacecolor='none', markeredgewidth=2, markersize=10, label='Error'),
    ]
    if centroids is not None:
        legend_elements.append(mlines.Line2D([], [], color='white', marker='X', markerfacecolor='black', markeredgecolor='white', markersize=15, label='Centroid'))

    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()