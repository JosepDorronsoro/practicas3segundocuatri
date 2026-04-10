from sklearn.datasets import make_blobs, make_moons, make_circles

def generate_gaussian_blobs():
    """Generates a synthetic dataset with spherical distributions (blobs).

    This dataset is ideal for centroid-based algorithms like K-means, 
    as the groups are convex and have controlled variance.

    Returns:
        tuple: A pair (X, y) where:
            - X (numpy.ndarray): Feature matrix of shape (500, 2).
            - y (numpy.ndarray): True cluster labels for validation.
    """
    # Standard clusters with different standard deviations
    X, y = make_blobs(
        n_samples=500, 
        centers=4, 
        cluster_std=[1.3, 3.5, 1.5, 2.0], 
        random_state=42
    )
    return X, y

def generate_moons_data():
    """Generates a synthetic dataset with two interlocking half-moon shapes.

    This dataset represents a classic failure scenario for K-means due to its 
    non-convex geometry, making it useful for introducing density-based models.

    Returns:
        tuple: A pair (X, y) where:
            - X (numpy.ndarray): Feature matrix of shape (500, 2).
            - y (numpy.ndarray): True cluster labels for validation.
    """
    X, y = make_moons(
        n_samples=500, 
        noise=0.2, 
        random_state=42
    )
    return X, y

def generate_circles_data(n_samples=500, noise=0.05, factor=0.5):
    """Generates a dataset of two concentric circles (one inside the other).

    This is another non-linearly separable dataset used to test 
    spectral clustering or algorithms that use the kernel trick.

    Args:
        n_samples (int): Total number of points.
        noise (float): Standard deviation of Gaussian noise added to the data.
        factor (float): Scale factor between the inner and outer circle (0 < factor < 1).

    Returns:
        tuple: (X, y) where X are the coordinates and y are class labels (0: outer, 1: inner).
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    return X, y