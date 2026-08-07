#!/usr/bin/env python3
"""
models/hms.py
-------------
Differentiable HMS Forward Operator H(x) and Radiance Innovation Loss Engine
for AIDA GNN Surrogate Model Training.

Computes simulated brightness temperatures (T_b) for a 12-channel HMS sounder
across 32 model vertical levels using log-pressure Gaussian weighting functions,
and calculates channel-weighted innovation residuals J_rad_hms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# HMS 12-Channel Sounder Configuration
# Channels 1-12, peak log-pressure locations (Pa), vertical widths (sigma_ln_p),
# and nominal sensor observation errors (Kelvin).
# -----------------------------------------------------------------------------
HMS_CHANNELS = list(range(1, 13))

# Approximate peak pressures (Pa) for channels 1-12
HMS_PEAK_P_PA = np.array([
    98000.0, 88000.0, 70000.0, 48000.0, 30000.0, 19000.0,
    12000.0, 7000.0,  3500.0,  1800.0,  800.0,   300.0
], dtype=np.float32)

# Channel Gaussian weighting widths in log-pressure space
HMS_SIGMA_LN_P = np.array([
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35
], dtype=np.float32)

# Nominal observation errors per channel (Kelvin)
HMS_CHAN_ERRORS = np.array([
    2.00, 1.50, 0.80, 0.50, 0.35, 0.30,
    0.30, 0.40, 0.60, 0.90, 1.40, 2.20
], dtype=np.float32)


class DifferentiableHMSOperator(nn.Module):
    """
    Differentiable Forward Radiance Operator H_hms(x) in PyTorch.
    Maps atmospheric profile states (Temperature T, Pressure p) to simulated HMS
    brightness temperatures T_b for 12 channels.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(HMS_CHANNELS)

        # Register channel parameters as non-trainable buffers
        self.register_buffer("peak_p", torch.from_numpy(HMS_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(HMS_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(HMS_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp_k: Temperature profile tensor [Batch, Levels=32, Nodes] (Kelvin)
            press_pa: Pressure profile tensor [Batch, Levels=32, Nodes] (Pascal)

        Returns:
            Simulated Brightness Temperatures [Batch, Channels=12, Nodes] (Kelvin)
        """
        press_pa = torch.clamp(press_pa, min=10.0)
        ln_press = torch.log(press_pa)  # [B, 32, N]

        ln_peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)  # [1, 12, 1, 1]
        sigma = self.sigma_ln_p.view(1, self.num_channels, 1, 1)

        ln_p_exp = ln_press.unsqueeze(1)  # [B, 1, 32, N]
        temp_exp = temp_k.unsqueeze(1)    # [B, 1, 32, N]

        # Log-pressure Gaussian weighting kernel [B, 12, 32, N]
        weight_logits = -0.5 * ((ln_p_exp - ln_peak_p) / sigma) ** 2
        weights = F.softmax(weight_logits, dim=2)  # Normalize across vertical levels

        # Integrate temperature profile: T_b = sum_k (T_k * W_k)
        tb_sim = torch.sum(temp_exp * weights, dim=2)  # [B, 12, N]
        return tb_sim


class HMSRadianceLoss(nn.Module):
    """
    Computes normalized HMS satellite radiance innovation loss J_rad_hms.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.operator = DifferentiableHMSOperator(num_levels=num_levels)

    def forward(
        self,
        pred_temp_k: torch.Tensor,
        pred_press_pa: torch.Tensor,
        obs_tb: torch.Tensor,
        obs_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred_temp_k: Predicted Temperature [Batch, 32, Nodes] (K)
            pred_press_pa: Predicted Pressure [Batch, 32, Nodes] (Pa)
            obs_tb: Observed HMS Brightness Temp [Batch, 12, Nodes] (K)
            obs_mask: Active FOV spatial mask [Batch, 12, Nodes] (1.0 = valid)

        Returns:
            Scalar loss tensor J_rad_hms
        """
        tb_sim = self.operator(pred_temp_k, pred_press_pa)  # [Batch, 12, Nodes]

        sigma = self.operator.obs_errors.view(1, -1, 1)  # [1, 12, 1]
        diff_normalized = (tb_sim - obs_tb) / sigma      # [Batch, 12, Nodes]

        if obs_mask is not None:
            sq_err = (diff_normalized ** 2) * obs_mask
            denom = torch.sum(obs_mask) + 1e-6
            return torch.sum(sq_err) / denom
        else:
            return torch.mean(diff_normalized ** 2)
