import numpy as np

def gaussian_kernel(u):
    """
    Multivariate Gaussian kernel.
    """
    d = u.shape[-1]
    return (1 / (2*np.pi)**(d/2)) * np.exp(-0.5 * np.sum(u**2, axis=-1))

def parzen_density(X_train, h=None, kernel=gaussian_kernel):
    """
    Fit a Parzen window density estimator.

    Parameters
    ----------
    X_train : array (N, d)
        Training data
    h : float
        Bandwidth
    kernel : function
        Kernel function

    Returns
    -------
    density : callable
        Function that evaluates the estimated density
    """

    N, d = X_train.shape
    
    def density(x):
        
        """
        Estimate the probability density at the given points using kernel density estimation.

        Args:
            x (array-like): Input points where the density should be evaluated.
                Shape can be (n_samples, n_features) or (n_features,).

        Returns:
            np.ndarray: Estimated density values for each input point.
        """
        
        return (1/(h**d)) * np.sum(kernel( (1/h) * (x[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ), axis=1)

    return density