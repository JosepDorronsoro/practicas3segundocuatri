# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:46:22 2026

@author: Iván Domínguez <ivan.dominguezh@estudiante.uam.es>
"""

import numpy as np
from numpy.linalg import eigh, inv
from typing import Union

def gaussian_pdf(
        x: Union[float, np.ndarray], 
        mu: float, 
        sigma: float
    ) -> Union[float, np.ndarray]:
    
    """
    Exponential probability density function.

    Args:
        x: Point or array of points at which to evaluate the pdf. 
        mu: expectation of X. 
        sigma: standard deviation of X. 

    Returns:
        Value of the exponential pdf evaluated at `x`.

    Examples:
    
        >>> 
        
        >>> 
        
        >>> 
        
        >>> 
        
    """
    
    if sigma <= 0:
        raise ValueError("Standard deviation must be positive.")
    
    x = np.asarray(x)
    
    return (1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / 2*sigma**2)

def multivariate_gaussian_pdf(
    x: Union[float, np.ndarray], 
    mu: np.ndarray[float], 
    sigma: np.ndarray[float]) -> Union[float, np.ndarray]:
    
    """
    Exponential probability density function.

    Args:
        x: Point or array of points at which to evaluate the pdf. 
        mu: expectation of X. 
        sigma: standard deviation of X. 

    Returns:
        Value of the exponential pdf evaluated at `x`.

    Examples:
    
        >>> 
        
        >>> 
        
        >>> 
        
        >>> 
        
    """
    
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Compute det(Sigma) using eigenvectors:
    eigenvalues, _ = eigh(sigma)
    det_sigma = np.prod(eigenvalues)
    
    
    d = x - mu
    
    exponent = np.sum((d @ inv(sigma)) * d, axis=1)
    norm = (1/np.sqrt((2*np.pi)**len(mu)*det_sigma))
    
    return norm * np.exp(-0.5 * exponent)