import numpy as np

def generate_regression_data(n_samples, dim, noise=0.1, random_state=None):
    """
    Generate moderately complex synthetic regression data.

    The target variable is created from a nonlinear combination of the
    input features including sinusoidal, polynomial, and interaction terms.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    dim : int
        Number of input features
    noise : float
        Standard deviation of the Gaussian noise added to the target
    random_state : int or None
        Random seed for reproducibility

    Returns
    -------
    X : array of shape (n_samples, dim)
        Input features
    y : array of shape (n_samples,)
        Target regression values
    """

    rng = np.random.default_rng(random_state)

    # Generate input features uniformly
    X = rng.uniform(-3, 3, size=(n_samples, dim))

    # Nonlinear regression function
    y = (
        np.sin(X[:, 0]) +
        0.5 * X[:, 0]**2
    )

    if dim > 1:
        y += np.cos(X[:, 1])

    if dim > 2:
        y += 0.5 * X[:, 0] * X[:, 2]

    if dim > 3:
        y += np.sin(X[:, 3] * X[:, 1])

    if dim > 4:
        y += 0.3 * np.sum(X[:, 4:]**2, axis=1)

    # Add Gaussian noise
    y += noise * rng.normal(size=n_samples)

    return X, y

    