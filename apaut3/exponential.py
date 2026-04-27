# -*- coding: utf-8 -*-
"""Exponential noise schedule for VP diffusion."""

import math
import torch

from .base import NoiseSchedule


class ExponentialSchedule(NoiseSchedule):
    """Exponential (geometric) noise schedule.

    Defines β(t) growing geometrically between β_min and β_max:

        β(t) = β_min · (β_max / β_min)^t = β_min · exp(k · t),
        with  k = ln(β_max / β_min)  and  t ∈ [0, 1].

    Equivalent — after the reparametrisation τ = (t − 1)/(T − 1) — to the
    discrete schedule introduced in the literature:

        β_t = β_min · (β_max / β_min)^((t − 1)/(T − 1)),   t ∈ [1, T].

    Boundary values
    ---------------
        β(0) = β_min,     β(1) = β_max.

    Cumulative integral
    -------------------
    Required by ``VPProcess`` to build α_t = exp(−B(t)) and σ_t = √(1 − α_t):

        B(t) = ∫₀ᵗ β(s) ds = (β(t) − β_min) / k.

    Compared to ``LinearSchedule`` and ``CosineSchedule``, the exponential
    schedule injects noise more aggressively at later timesteps (its
    derivative β'(t) = k · β(t) grows with β itself), which can be useful
    when one wants a stronger denoising signal in the high-noise regime.

    Args:
        beta_min: β at t = 0.   Must be > 0.   Default 0.1.
        beta_max: β at t = 1.   Must be > 0 and ≠ β_min.   Default 20.0.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be strictly positive.")
        if beta_max == beta_min:
            raise ValueError(
                "beta_max must differ from beta_min for an exponential schedule."
            )
        self.beta_min = beta_min
        self.beta_max = beta_max
        # k = ln(β_max / β_min)  — precomputed for numerical efficiency
        self._k = math.log(beta_max / beta_min)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # β(t) = β_min · exp(k · t)
        return self.beta_min * torch.exp(self._k * t)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        # B(t) = (β(t) − β_min) / k  =  β_min · (exp(k · t) − 1) / k
        return (self.beta(t) - self.beta_min) / self._k
