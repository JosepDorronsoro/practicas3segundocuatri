# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:03:48 2026

@author: Alberto Suárez <alberto.suarez@uam.es>
"""

import numpy as np
from scipy.optimize import minimize
from typing import Union

from typing import Callable


def maximum_likelihood_fit(
        pdf: Callable, 
        X: np.ndarray
    ) -> float:
    
    # [TODO]: Doctrings (including doctests) and code 
    
    """
    Estimates the lambda parameter of an exponential distribution using MLE.

    Args:
        pdf: The probability density function.
        X: Array of data points.

    Returns:
        The estimated lambda parameter.

    Examples:
        >>> np.random.seed(42)
        >>> X = np.random.exponential(scale=0.5, size=1000)
        >>> dummy_pdf = lambda x, lam: x 
        >>> lam_mle = maximum_likelihood_fit(dummy_pdf, X)
        >>> np.isclose(lam_mle, 2.0, atol=0.2)
        True
    """

    # In order to transform the original X we assume lambda=1.5.    
    lam_true = 1.5

    # We vectorize de original X according to the cdf:
    vectorize_exp = np.vectorize(pdf)
    X_exp = vectorize_exp(X, lam=lam_true)
    
    # As we will maximize the log-likelihood, we define the negative
    # log-likelihood in order to minimize it. 
    def negative_likelihood(lam, X_exp):
        ''' Returns the negative likelihood in terms of lambda'''
        return - (len(X)*np.log(lam) - lam*np.sum(X_exp))
    
    # We minimize the negative-log-likelihood with X_exp as parameter.
    return minimize(negative_likelihood, x0=1, args=(X_exp)).x[0]


def maximum_posterior_fit(
        pdf: Callable,
        X: np.ndarray, 
        alpha: Union[float,int],
        beta: Union[float,int],
    ) -> float:
    
    # [TODO]: Doctrings (including doctests) and code 
    
    """
    Estimates the lambda parameter using Maximum A Posteriori (MAP).

    Args:
        pdf: The probability density function.
        X: Array of data points.
        alpha: Shape parameter of the Gamma prior.
        beta: Rate parameter of the Gamma prior.

    Returns:
        The estimated lambda parameter (MAP).

Examples:
        >>> X = np.array([0.5, 0.5, 0.8, 0.2])
        >>> dummy_pdf = lambda x, lam: x
        >>> res = maximum_posterior_fit(dummy_pdf, X, alpha=2, beta=1)
        >>> res > 0
        True
    """
    
    lam_true = 1.5
    
    vectorize_exp = np.vectorize(pdf)
    X_exp = vectorize_exp(X, lam=lam_true)
    
    def negative_posterior(lam, X_exp, alpha, beta):
        return - ( len(X_exp)*np.log(lam) - lam*np.sum(X_exp) - (alpha-1)*np.log(lam) + beta*lam)
    
    return minimize(negative_posterior, x0=1, args=(X_exp, alpha, beta)).x[0]

                     # this term comes from likelihood
                                                    # this term comes from prior

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)