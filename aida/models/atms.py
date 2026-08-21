#!/usr/bin/env python3
"""
models/atms.py
--------------
Differentiable ATMS Forward Operator H(x) and Radiance Innovation Loss Engine
for AIDA GNN Surrogate Model Training.

Computes simulated brightness temperatures (T_b) for all 22 ATMS channels
across 32 model vertical levels using log-pressure Gaussian weighting functions,
and calculates channel-weighted innovation residuals J_rad_atms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# ATMS 22-Channel Sounder Configuration
# Channels 1-22: Peak log-pressure locations (Pa), vertical widths (sigma_ln_p),
# and nominal sensor observation errors (Kelvin).
# -----------------------------------------------------------------------------
ATMS_CHANNELS = list(range(1, 23))

# Approximate peak pressures (Pa) for ATMS channels 1-22
ATMS_PEAK_P_PA = np.array([
    100000.0, 95000.0, 85000.0, 70000.0, 50000.0, 38000.0, 25000.0, 15000.0,
    8000.0,   5000.0,  2500.0,  1000.0,  500.0,   200.0,   100.0,   90000.0,
    80000.0,  60000.0, 45000.0, 30000.0, 20000.0, 12000.0
], dtype=np.float32)

# Channel Gaussian weighting widths in log-pressure space
ATMS_SIGMA_LN_P = np.array([
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    0.40, 0.40, 0.40, 0.40, 0.40, 0.40
], dtype=np.float32)

# Nominal observation errors per channel (Kelvin)
ATMS_CHAN_ERRORS = np.array([
    2.50, 2.20, 1.20, 0.60, 0.30, 0.25, 0.25, 0.25,
    0.25, 0.35, 0.55, 0.80, 1.20, 1.80, 3.50, 2.50,
    2.00, 1.50, 1.20, 1.00, 1.10, 1.30
], dtype=np.float32)


class DifferentiableATMSOperator(nn.Module):
    """
    Differentiable Forward Radiance Operator H_atms(x) in PyTorch.
    Maps atmospheric profile states (Temperature T, Pressure p) to simulated ATMS
    brightness temperatures T_b for 22 channels.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(ATMS_CHANNELS)

        self.register_buffer("peak_p", torch.from_numpy(ATMS_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(ATMS_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(ATMS_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp_k: Temperature profile tensor [Batch, Levels=32, Nodes] (Kelvin)
            press_pa: Pressure profile tensor [Batch, Levels=32, Nodes] (Pascal)

        Returns:
            Simulated Brightness Temperatures [Batch, Channels=22, Nodes] (Kelvin)
        """
        press_pa = torch.clamp(press_pa, min=10.0)
        ln_press = torch.log(press_pa)  # [B, 32, N]

        ln_peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)  # [1, 22, 1, 1]
        sigma = self.sigma_ln_p.view(1, self.num_channels, 1, 1)

        ln_p_exp = ln_press.unsqueeze(1)  # [B, 1, 32, N]
        temp_exp = temp_k.unsqueeze(1)    # [B, 1, 32, N]

        # Log-pressure Gaussian weighting kernel [B, 22, 32, N]
        weight_logits = -0.5 * ((ln_p_exp - ln_peak_p) / sigma) ** 2
        weights = F.softmax(weight_logits, dim=2)  # Normalize across vertical levels

        # Integrate temperature profile: T_b = sum_k (T_k * W_k)
        tb_sim = torch.sum(temp_exp * weights, dim=2)  # [B, 22, N]
        return tb_sim


class ATMSRadianceLoss(nn.Module):
    """Computes normalized ATMS satellite radiance innovation loss J_rad_atms."""
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.operator = DifferentiableATMSOperator(num_levels=num_levels)

    def forward(
        self,
        pred_temp_k: torch.Tensor,
        pred_press_pa: torch.Tensor,
        obs_tb: torch.Tensor,
        obs_mask: torch.Tensor = None
    ) -> torch.Tensor:
        tb_sim = self.operator(pred_temp_k, pred_press_pa)  # [Batch, 22, Nodes]

        sigma = self.operator.obs_errors.view(1, -1, 1)  # [1, 22, 1]
        diff_normalized = (tb_sim - obs_tb) / sigma      # [Batch, 22, Nodes]

        if obs_mask is not None:
            sq_err = (diff_normalized ** 2) * obs_mask
            denom = torch.sum(obs_mask) + 1e-6
            return torch.sum(sq_err) / denom
        else:
            return torch.mean(diff_normalized ** 2)
