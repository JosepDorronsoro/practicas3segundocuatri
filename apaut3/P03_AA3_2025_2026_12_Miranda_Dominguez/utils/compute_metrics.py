from numpy import clip, mean, log, abs, sum

def compute_metrics(true_density, est_density, sample_from_true_density, dim, n_eval=2000):
    """
    Compute several divergence and error metrics between a true density
    function and an estimated density function.

    Args:
        true_density (callable): Function that returns the true density values
            for a given set of input points.
        est_density (callable): Function that returns the estimated density values
            for a given set of input points.
        sample_from_true_density (callable): Function used to generate samples
            from the true density distribution.
        dim (int): Dimensionality of the sampled data.
        n_eval (int, optional): Number of evaluation samples to generate.
            Defaults to 2000.

    Returns:
        tuple:
            - mse (float): Mean squared error between true and estimated densities.
            - kl (float): Kullback-Leibler divergence.
            - js (float): Jensen-Shannon divergence.
            - tv (float): Total variation distance.
    """

    X = sample_from_true_density(true_density, dim, n_eval)

    p_true = true_density(X)
    p_est = est_density(X)

    eps = 1e-12
    p_true = clip(p_true, eps, None)
    p_est = clip(p_est, eps, None)

    mse = mean((p_true - p_est)**2)

    kl = mean(p_true * log(p_true / p_est))

    m = 0.5 * (p_true + p_est)
    js = 0.5*mean(p_true*log(p_true/m)) + 0.5*mean(p_est*log(p_est/m))

    tv = 0.5 * mean(abs(p_true - p_est))

    return mse, kl, js, tv
