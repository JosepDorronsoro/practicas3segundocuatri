# -*- coding: utf-8 -*-
"""Imputation sampler: inpaint missing image regions given observed pixels."""

import numpy as np
import torch
import torch.nn as nn


class ImputationSampler:
    """Generates missing image regions conditioned on observed pixels.

    At each denoising step the algorithm:
    1. Takes a reverse Euler-Maruyama step on the full image.
    2. Replaces the *observed* pixels with the correctly-noised original,
       so the generated region stays consistent with the known region.

    This "replacement" strategy follows the RePaint approach (Lugmayr et al., 2022).

    Args:
        n_resample: Number of forward/backward resampling cycles per step.
                    Values > 1 improve coherence at the boundary between known
                    and unknown regions.  Default 1.
    """

    def __init__(self, n_resample: int = 1):
        self.n_resample = n_resample

    @torch.no_grad()
    def inpaint(
        self,
        score_model: nn.Module,
        process,
        x_known: torch.Tensor,
        mask: torch.Tensor,
        n_steps: int = 500,
        device: torch.device = torch.device("cpu"),
        T: float = 1.0,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Inpaint the unknown pixels of x_known.

        Args:
            score_model: Trained score network s_θ(x, t).
            process:     A DiffusionProcess instance.
            x_known:     Original image [B, C, H, W] in [0, 1].
                         Unknown pixels can be any value — they are ignored.
            mask:        Binary mask [B, 1, H, W].
                         1 = observed pixel, 0 = pixel to inpaint.
            n_steps:     Number of reverse integration steps.
            device:      Torch device.
            T, eps:      Integration interval [eps, T].

        Returns:
            Inpainted image tensor [B, C, H, W].
        """
        B, C, H, W = x_known.shape
        img_shape = (C, H, W)
        view_shape = [B] + [1] * len(img_shape)

        x_known = x_known.to(device)
        mask = mask.to(device).float()

        x = process.prior_sample((B, *img_shape), device)
        dt = (eps - T) / n_steps
        t_vals = torch.linspace(T, eps, n_steps + 1, device=device)

        score_model.eval()
        for i in range(n_steps):
            t_cur = t_vals[i].expand(B)
            t_nxt = t_vals[i + 1].expand(B)

            for _ in range(self.n_resample):
                # ── Reverse step on the full image ─────────────────────────
                score = score_model(x, t_cur)
                drift = process.reverse_drift(x, t_cur, score)
                g_t = process.diffusion_coefficient(t_cur).view(*view_shape)
                z = torch.randn_like(x)
                x_pred = x + drift * dt + g_t * np.sqrt(abs(dt)) * z

                # ── Replace observed pixels with correctly-noised original ─
                x_known_noisy, _ = process.perturb(x_known, t_nxt)
                x = mask * x_known_noisy + (1.0 - mask) * x_pred

        return x
