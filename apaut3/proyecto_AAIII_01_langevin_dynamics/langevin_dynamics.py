# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:08:56 2024

@author: alberto.suarez@uam.es
"""
from __future__ import annotations

from typing import Callable, Union
from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def simulate_langevin_dynamics_euler_maruyama(
        x_0: ArrayLike,
        t_end: float,
        n_steps: int,
        score_function: Callable, 
        diffusion: Callable,
        seed: Union[int, None] = None,
    ):
    """ Simulate Langevin dynamics using the Euler-Maruyama integrator 
    
     Args:
        x_0: The initial points of the trajectory

        score_function: Stein's score function
        diffusion_term: Function of :math`(x(t),t)` defining the diffusion term
        t_end: endpoint of the integration interval    
        n_steps: number of integration steps 
            
    Returns:
        x_t: Trajectories that result from the integration of the SDE.
             The shape is (*np.shape(x_0), (n_steps + 1))
            
    Notes:
        The implementation is fully vectorized except for a loop over time.

    Example 1:
        
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> score_function = lambda x_t, t: - x_t
        >>> diffusion = lambda t: 2.0
        >>> x_0 = 6.3
        >>> t_end = 5.0
        >>> n_steps = 1000
        >>> times, x_t = simulate_langevin_dynamics_euler_maruyama(
        ...     x_0, t_end, n_steps, score_function, diffusion
        ... )
        >>> _ = plt.plot(times, x_t)
        >>> print(times[:5])
        [0.    0.005 0.01  0.015 0.02 ]
        >>> print(np.shape(x_t))
        (1001,)
  
        
    Example 2:
        
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> score_function = lambda x_t, t: - x_t
        >>> diffusion = lambda t: 2.0
        >>> n_trajectories = 10
        >>> x_0 = 6.3 * np.ones(n_trajectories) 
        >>> t_end = 5.0
        >>> n_steps = 1000
        >>> times, x_t = simulate_langevin_dynamics_euler_maruyama(
        ...     x_0, t_end, n_steps, score_function, diffusion
        ... )
        >>> _ = plt.plot(times, x_t.T)
        >>> print(times[:5])
        [0.    0.005 0.01  0.015 0.02 ]
        >>> print(np.shape(x_t))
        (10, 1001)

    Example 3:
        
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> score_function = lambda x_t, t: - x_t
        >>> diffusion = lambda t: 2.0
        >>> n_trajectories = 10
        >>> x_0 = 4.0 + 10.0 * np.random.rand(n_trajectories)
        >>> t_end = 5.0
        >>> n_steps = 1000
        >>> times, x_t = simulate_langevin_dynamics_euler_maruyama(
        ...     x_0, t_end, n_steps, score_function, diffusion
        ... )
        >>> _ = plt.plot(times, x_t.T)
        >>> print(times[:5])
        [0.    0.005 0.01  0.015 0.02 ]
        >>> print(np.shape(x_t))
        (10, 1001)

    """
  
    
    times = np.linspace(0.0, t_end, n_steps + 1)
    dt = times[1] - times[0]
    
    x_t = np.empty((*np.shape(x_0), n_steps + 1))
    
    x_t[..., 0] = np.asarray(x_0)
    
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(np.shape(x_t))

    for n, t in enumerate(times[:-1]):
        diffusion_term = diffusion(t)
        x_t[..., n + 1] = (
            x_t[..., n]
            + 0.5 * diffusion_term**2 * score_function(x_t[..., n], t) * dt 
            + diffusion_term * np.sqrt(dt) * z[..., n]
        )
        
    return times, x_t

def animation_pdf_discrete(
        x,
        p_t,
        T,
        model_pdf=None,
        y_max=None,
        interval=10,
    ): 
    # Create a figure and axes.
    fig, ax = plt.subplots()
    delta_x = x[1] - x[0]
    x_min = x[0] - 0.5 * delta_x
    x_max = x[-1] + 0.5 * delta_x
    ax.set_xlim((x_min, x_max))
    
    if y_max == None:
        y_max = 1.2 * np.max(p_t[:])

    ax.set_ylim((0.0, y_max))

    # Create objects that change in the animiation.
    bars = ax.bar(x, p_t[:, 0], width=(0.8 * delta_x))  
    
    if model_pdf:
        x_plot = np.linspace(x_min, x_max, 1000)
        ax.plot(x_plot, model_pdf(x_plot), color='r', linewidth=3)
    
    
    # Animation function. This function is called sequentially.
    def drawframe(t):
        for i, artist in enumerate(bars):
            artist.set_height(p_t[i, t]) 
        return bars

    return ( 
        fig, 
        ax, 
        animation.FuncAnimation(fig, drawframe,  frames=T, interval=interval)
    )


def countour_plot_force_field(
        X, 
        pdf, 
        score_function, 
        n_steps, 
        x_range, 
        y_range
    ):
    n_plot = 2**6
    n_grid = 2**4
    factor_plot_grid = n_plot // n_grid
    
    x_plot = np.linspace(*x_range, n_plot)
    y_plot = np.linspace(*y_range, n_plot)
    
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    X_grid = X_plot[::factor_plot_grid, ::factor_plot_grid]
    Y_grid = Y_plot[::factor_plot_grid, ::factor_plot_grid]
    
    force_field = score_function(
        np.column_stack((np.ravel(X_grid), np.ravel(Y_grid)))
    )
    
    fig, ax = plt.subplots()
    
    _ = ax.quiver(
        X_grid, 
        Y_grid, 
        force_field[:, 0], 
        force_field[:, 1],
        alpha=0.7,
    )
    
    Z_plot = pdf(np.column_stack((np.ravel(X_plot), np.ravel(Y_plot))))
    Z_plot = np.reshape(Z_plot, np.shape(X_plot))
    
    _ = ax.contour(X_plot, Y_plot, np.log(Z_plot), levels=25)
    
    scatter_plot = ax.scatter(X[:, 0, 0], X[:, 1, 0], color='r', s=10.0)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    def drawframe(n):
        scatter_plot.set_offsets(X[:, :, n])
        return scatter_plot
     
    return ( 
        fig, 
        ax, 
        animation.FuncAnimation(
            fig, 
            drawframe,  
            frames=(n_steps + 1), 
            interval=1,
        )
    )

if __name__ == "__main__":
    import doctest
    doctest.testmod()