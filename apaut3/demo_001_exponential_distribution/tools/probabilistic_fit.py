# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:03:48 2026

@author: Alberto Suárez <alberto.suarez@uam.es>
"""

import numpy as np
from scipy.optimize import minimize

from typing import Callable


def maximum_likelihood_fit(
        pdf: Callable, 
        X: np.ndarray
    ) -> float:
    # [TODO]: Doctrings (including doctests) and code 
    pass



def maximum_posterior_fit(
        pdf: Callable,
        prior: Callable,
        X: np.ndarray
    ) -> float:
    # [TODO]: Doctrings (including doctests) and code 
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
