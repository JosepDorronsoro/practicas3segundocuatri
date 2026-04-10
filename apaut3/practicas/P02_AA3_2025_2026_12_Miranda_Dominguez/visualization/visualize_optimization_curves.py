import matplotlib.pyplot as plt

def visualize_optimization_curves(k_list, inertias, silhouette_scores):
    """
    Generates a visual comparison of metrics for selecting the optimal k.

    Args:
        k_list (range/list): Range of evaluated k values.
        inertias (list): Resulting inertia values (Within-Cluster Sum of Squares).
        silhouette_scores (list): Resulting silhouette scores.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow Method Plot (Inertia)
    ax1.plot(k_list, inertias, marker='o', linestyle='-', color='b')
    ax1.set_title("Elbow Method (Inertia)", fontsize=14)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (SSE)")
    ax1.grid(True, alpha=0.3)

    # Silhouette Analysis Plot
    ax2.plot(k_list, silhouette_scores, marker='s', linestyle='-', color='g')
    ax2.set_title("Silhouette Analysis", fontsize=14)
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()