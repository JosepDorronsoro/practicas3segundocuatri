# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:46:22 2026

@author: Alberto Suárez <alberto.suarez@uam.es>
"""

import numpy as np
from typing import Union

def exp_pdf(
        x: Union[float, np.ndarray], 
        lam: float
    ) -> Union[float, np.ndarray]:
    """
    Exponential probability density function.

    Args:
        x: Point or array of points at which to evaluate the pdf. 
           Must satisfy x >= 0.
        lam: Rate parameter λ > 0.0

    Returns:
        Value of the exponential pdf evaluated at `x`.

    Examples:
        >>> exp_pdf(0.0, 2.0)
        2.0
        >>> exp_pdf(1.0, 2.0)
        0.2706705664732254
        >>> exp_pdf(np.array([-1.0, 0.0, 1.0]), 1.0)
        array([0.        , 1.        , 0.36787944])
    """

    return lam * np.exp(-lam*x)


def exp_cdf(
        x: Union[float, np.ndarray], 
        lam: float
    ) -> Union[float, np.ndarray]:
    """
    Exponential cumulative distribution function.

    Args:
        x: Point or array of points at which to evaluate the pdf. 
           Must satisfy x >= 0.
        lam: Rate parameter λ > 0.0

    Returns:
        Value of the exponential cdf evaluated at `x`.

    Examples:
        >>> exp_cdf(0.0, 2.0)
        0.0
        >>> exp_cdf(1.0, 2.0)
        0.8646647167633873
        >>> exp_cdf(np.array([-1.0, 0.0, 1.0]), 1.0)
        array([0.        , 0.        , 0.63212056])
    """

    return 1 - np.exp(-lam*x)


def exp_inv(
        p: Union[float, np.ndarray], 
        lam: float
    ) -> Union[float, np.ndarray]:
    """
    Inverse CDF (quantile function) of the exponential distribution.

    Args:
        u: Probability value(s) in the open interval (0, 1).
        lam: Rate parameter λ > 0.

    Returns:
        Quantile corresponding to probability `u`.

    Examples:
        >>> exp_inv(0.5, 2.0)
        0.34657359037935253
        >>> exp_inv(np.array([0.25, 0.5]), 1.0)
        array([0.28768207, 0.69314718])
        >>> np.allclose(exp_cdf(exp_inv(0.3, 1.5), 1.5), 0.3)
        True
    """

    return (-1/lam) * np.log(p) 


if __name__ == "__main__":
    import doctest
    doctest.testmod()
