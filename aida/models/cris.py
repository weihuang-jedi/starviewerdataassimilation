#!/usr/bin/env python3
"""
models/cris.py
--------------
Differentiable CrIS Forward Operator H(x) and Radiance Innovation Loss Engine
for AIDA GNN Surrogate Model Training.

Computes simulated brightness temperatures (T_b) for a 30-channel CrIS subset
across 32 model vertical levels using log-pressure Gaussian weighting functions,
and calculates channel-weighted innovation residuals J_rad_cris.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# CrIS 30-Channel DA Subset Configuration
# Representative channels covering CO2 longwave temperature, surface window,
# moisture, and shortwave temperature sounding bands.
# -----------------------------------------------------------------------------
CRIS_CHANNELS = [
    12, 28, 44, 62, 85, 102, 125, 150, 182, 210,
    240, 275, 310, 355, 400, 480, 520, 600, 720, 850,
    980, 1050, 1120, 1200, 1280, 1350, 1420, 1500, 1580, 1650
]

# Approximate peak pressures (Pa) for channels 1-30
CRIS_PEAK_P_PA = np.array([
    250.0, 600.0, 1200.0, 3000.0, 6000.0, 10000.0, 18000.0, 28000.0, 40000.0, 55000.0,
    75000.0, 88000.0, 98000.0, 100000.0, 95000.0, 30000.0, 100000.0, 25000.0, 95000.0,
    82000.0, 62000.0, 48000.0, 38000.0, 28000.0, 20000.0, 14000.0, 9000.0,
    6000.0, 2500.0, 1000.0
], dtype=np.float32)

# Channel Gaussian weighting widths in log-pressure space
CRIS_SIGMA_LN_P = np.array([
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    0.35, 0.35, 0.25, 0.20, 0.30, 0.40, 0.20, 0.40, 0.20,
    0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.40, 0.40,
    0.35, 0.35, 0.35
], dtype=np.float32)

# Nominal observation errors per channel (Kelvin)
CRIS_CHAN_ERRORS = np.array([
    1.40, 1.10, 0.85, 0.55, 0.40, 0.30, 0.28, 0.25, 0.25, 0.28,
    0.38, 0.48, 0.75, 1.15, 0.95, 0.85, 1.40, 1.05, 1.35,
    0.75, 0.65, 0.60, 0.55, 0.65, 0.80, 1.05, 1.25,
    1.45, 1.75, 2.10
], dtype=np.float32)


class DifferentiableCrISOperator(nn.Module):
    """
    Differentiable Forward Radiance Operator H_cris(x) in PyTorch.
    Maps atmospheric profile states (Temperature T, Pressure p) to simulated CrIS
    brightness temperatures T_b for 30 channels.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(CRIS_CHANNELS)

        self.register_buffer("peak_p", torch.from_numpy(CRIS_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(CRIS_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(CRIS_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp_k: Temperature profile tensor [Batch, Levels=32, Nodes] (Kelvin)
            press_pa: Pressure profile tensor [Batch, Levels=32, Nodes] (Pascal)

        Returns:
            Simulated Brightness Temperatures [Batch, Channels=30, Nodes] (Kelvin)
        """
        press_pa = torch.clamp(press_pa, min=10.0)
        ln_press = torch.log(press_pa)  # [B, 32, N]

        ln_peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)  # [1, 30, 1, 1]
        sigma = self.sigma_ln_p.view(1, self.num_channels, 1, 1)

        ln_p_exp = ln_press.unsqueeze(1)  # [B, 1, 32, N]
        temp_exp = temp_k.unsqueeze(1)    # [B, 1, 32, N]

        # Log-pressure Gaussian weighting kernel [B, 30, 32, N]
        weight_logits = -0.5 * ((ln_p_exp - ln_peak_p) / sigma) ** 2
        weights = F.softmax(weight_logits, dim=2)  # Normalize across vertical levels

        tb_sim = torch.sum(temp_exp * weights, dim=2)  # [B, 30, N]
        return tb_sim


class CrISRadianceLoss(nn.Module):
    """Computes normalized CrIS satellite radiance innovation loss J_rad_cris."""
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.operator = DifferentiableCrISOperator(num_levels=num_levels)

    def forward(
        self,
        pred_temp_k: torch.Tensor,
        pred_press_pa: torch.Tensor,
        obs_tb: torch.Tensor,
        obs_mask: torch.Tensor = None
    ) -> torch.Tensor:
        tb_sim = self.operator(pred_temp_k, pred_press_pa)  # [Batch, 30, Nodes]

        sigma = self.operator.obs_errors.view(1, -1, 1)  # [1, 30, 1]
        diff_normalized = (tb_sim - obs_tb) / sigma      # [Batch, 30, Nodes]

        if obs_mask is not None:
            sq_err = (diff_normalized ** 2) * obs_mask
            denom = torch.sum(obs_mask) + 1e-6
            return torch.sum(sq_err) / denom
        else:
            return torch.mean(diff_normalized ** 2)
