# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:46:22 2026

@author: Alberto Suárez <alberto.suarez@uam.es>
"""


import numpy as np
from typing import Union
from numpy.typing import ArrayLike

def gaussian_pdf(
    x: Union[float, ArrayLike], 
    mean: float = 0.0,
    stdev: float = 1.0,
) -> Union[float, ArrayLike]:
    """Evaluates the probability density function of a Gaussian distribution.

    Args:
        x: The value(s) at which to evaluate the PDF. 
        mean: The mean ($\mu$) of the distribution.
        stdev: The standard deviation ($\sigma$) of the distribution. 
               It must be positive. 

    Returns:
        The evaluation of the pdf. 
        It Returns a scalar np.float64 for a single input
        or a 1D array of shape (N,) for and array of N samples.
        
    Raises:
        ValueError: If `stdev` is less than or equal to 0.

    Examples:
    
        >>> # Standard normal 
        >>> round(gaussian_pdf(1.5), 4)
        np.float64(0.1295)
        
        >>> # Evaluate at a single point
        >>> round(gaussian_pdf(1.5, mean=-1.0, stdev=2.5), 4)
        np.float64(0.0968)
        
        >>> # Evaluate at a set of values
        >>> x_vals = [-1.0, 0.0, 1.5]
        >>> np.round(gaussian_pdf(x_vals, mean=-1.0, stdev=2.5), 4)
        array([0.1596, 0.1473, 0.0968])
    
    """
    if stdev <= 0:
        raise ValueError("Standard deviation must be positive.")
    
    x  = np.asarray(x)
    
    return (1/np.sqrt(2 * np.pi * stdev**2)) * np.exp(-(x-mean)**2 / 2*stdev**2)



def multivariate_gaussian_pdf(
    x: Union[float, ArrayLike], 
    mean_vector: float,
    covariance_matrix: float,
) -> Union[float, ArrayLike]:
    """Evaluates the pdf of a multivariate Gaussian distribution.

   Args:
        x: The vector or set of vectors to evaluate. 
           It can be a single input vector; that is, a 1D array of shape (D,) 
           or a 2D array of shape (N, D) for a sample of size N.
        mean: The mean vector of shape (D,). 
              It defaults to a zero vector.
        covariance: The covariance matrix of shape (D, D). 
                    It defaults to the identity matrix.

    Returns:
        The evaluation of the pdf. 
        It Returns a scalar np.float64 for a single input vector
        or a 1D array of shape (N,) for and array of N samples.

    Raises:
        ValueError: If the dimensions of x, mean_vector, and covariance_matrix
                    do not match.
        np.linalg.LinAlgError: If the covariance matrix is not invertible.

    Examples:

        >>> # 1D Gaussian: Evaluate at a single point
        >>> round(multivariate_gaussian_pdf(1.5, [-1.0], [[2.5**2]]), 4)
        np.float64(0.0968)
        
        >>> # 1D Gaussian: Evaluate at a set of values
        >>> x = [-1.0, 0.0, 1.5]
        >>> np.round(multivariate_gaussian_pdf(x, [-1.0], [[2.5**2]]), 4)
        array([0.1596, 0.1473, 0.0968])
        
        >>> # 2D Gaussian: Single point evaluation
        >>> x = [1.5, -2.5]
        >>> mu = [1.0, -2.0]
        >>> Sigma = [[1.0, 0.5], [0.5, 1.0]]
        >>> np.round(multivariate_gaussian_pdf(x, mu, Sigma), 4)
        np.float64(0.1115)

        >>> # 2D Gaussian: Data matrix
        >>> x = [[-1.0, -1.0], [0.0, 0.0], [2.5, -1.5]]
        >>> np.round(multivariate_gaussian_pdf(x, mu, Sigma), 4)
        array([0.0017, 0.0017, 0.0572])
        
    """
   
  
    x = np.asarray(x)
    mean_vector = np.asarray(mean_vector)
    covariance_matrix = np.asarray(covariance_matrix)
    
    D = len(mean_vector)
    n_rows, n_cols = np.shape(covariance_matrix)
    
    if n_rows != D or n_cols != D:
        raise ValueError(
            "The dimensions of mean and covariance matrix do not match."
        )
  
    # Ensure x is an array of dimension (N, D) 
    if x.ndim == 0:
        if D == 1:
            x = np.reshape(x, (1, 1))
        else: 
            raise ValueError("The dimensions of x do not match.")
    
    elif x.ndim == 1: 
        if D == 1:
            x = x[:, np.newaxis]
        elif len(x) == D:
            x = x[np.newaxis, :]            
        else:
            raise ValueError("The dimensions of x do not match.")
            
    N, n_columns = np.shape(x)
    
    if n_columns != D:
        raise ValueError("The dimensions of x do not match.")
  
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
  
    # [TODO: Complete code]
    
    pass  

if __name__ == "__main__":
    import doctest
    doctest.testmod()