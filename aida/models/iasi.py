#!/usr/bin/env python3
"""
models/iasi.py
--------------
Differentiable IASI Forward Operator H(x) and Radiance Innovation Loss Engine
for AIDA GNN Surrogate Model Training.

Computes simulated brightness temperatures (T_b) for a representative 30-channel
IASI subset across 32 model vertical levels using log-pressure Gaussian weighting functions,
and calculates channel-weighted innovation residuals J_rad_iasi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# IASI 30-Channel DA Subset Configuration
# Channel numbers, peak log-pressure locations (Pa), vertical widths (sigma_ln_p),
# and nominal sensor observation errors (Kelvin).
# -----------------------------------------------------------------------------
IASI_CHANNELS = [
    16, 39, 49, 106, 122, 145, 180, 212, 236, 249,
    275, 306, 345, 386, 404, 523, 921, 1027, 1194,
    1427, 1585, 1643, 1766, 2119, 2321, 2742, 2993,
    3014, 3217, 3580
]

# Approximate peak pressures (Pa) for channels 1-30
IASI_PEAK_P_PA = np.array([
    200.0, 500.0, 1000.0, 2500.0, 5000.0, 8000.0, 15000.0, 25000.0, 38000.0, 50000.0,
    70000.0, 85000.0, 95000.0, 100000.0, 98000.0, 35000.0, 100000.0, 20000.0, 98000.0,
    80000.0, 60000.0, 45000.0, 35000.0, 25000.0, 18000.0, 12000.0, 8000.0,
    5000.0, 2000.0, 800.0
], dtype=np.float32)

# Channel Gaussian weighting widths in log-pressure space
IASI_SIGMA_LN_P = np.array([
    0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    0.35, 0.35, 0.25, 0.20, 0.30, 0.40, 0.20, 0.40, 0.20,
    0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.40, 0.40,
    0.35, 0.35, 0.35
], dtype=np.float32)

# Nominal observation errors per channel (Kelvin)
IASI_CHAN_ERRORS = np.array([
    1.50, 1.20, 0.90, 0.60, 0.45, 0.35, 0.30, 0.25, 0.25, 0.30,
    0.40, 0.50, 0.80, 1.20, 1.00, 0.90, 1.50, 1.10, 1.40,
    0.80, 0.70, 0.65, 0.60, 0.70, 0.85, 1.10, 1.30,
    1.50, 1.80, 2.20
], dtype=np.float32)


class DifferentiableIASIOperator(nn.Module):
    """
    Differentiable Forward Radiance Operator H_iasi(x) in PyTorch.
    Maps atmospheric profile states (Temperature T, Pressure p) to simulated IASI
    brightness temperatures T_b for 30 channels.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(IASI_CHANNELS)

        # Register channel parameters as non-trainable buffers
        self.register_buffer("peak_p", torch.from_numpy(IASI_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(IASI_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(IASI_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp_k: Temperature profile tensor [Batch, Levels=32, Nodes] (Kelvin)
            press_pa: Pressure profile tensor [Batch, Levels=32, Nodes] (Pascal)

        Returns:
            Simulated Brightness Temperatures [Batch, Channels=30, Nodes] (Kelvin)
        """
        # Ensure positive pressure for log-space calculation
        press_pa = torch.clamp(press_pa, min=10.0)
        ln_press = torch.log(press_pa)  # [B, 32, N]

        ln_peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)  # [1, 30, 1, 1]
        sigma = self.sigma_ln_p.view(1, self.num_channels, 1, 1)

        # Reshape for broadcasting: [B, 1, Levels=32, Nodes]
        ln_p_exp = ln_press.unsqueeze(1)  # [B, 1, 32, N]
        temp_exp = temp_k.unsqueeze(1)    # [B, 1, 32, N]

        # Compute log-pressure Gaussian weighting kernel
        # Shape: [B, 30, 32, N]
        weight_logits = -0.5 * ((ln_p_exp - ln_peak_p) / sigma) ** 2
        weights = F.softmax(weight_logits, dim=2)  # Normalize across vertical levels

        # Integrate temperature profile across levels: T_b = sum_k (T_k * W_k)
        tb_sim = torch.sum(temp_exp * weights, dim=2)  # [B, 30, N]
        return tb_sim


class IASIRadianceLoss(nn.Module):
    """
    Computes normalized IASI satellite radiance innovation loss J_rad_iasi.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.operator = DifferentiableIASIOperator(num_levels=num_levels)

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
            obs_tb: Observed IASI Brightness Temp [Batch, 30, Nodes] (K)
            obs_mask: Active FOV spatial mask [Batch, 30, Nodes] (1.0 = valid, 0.0 = masked)

        Returns:
            Scalar loss tensor J_rad_iasi
        """
        tb_sim = self.operator(pred_temp_k, pred_press_pa)  # [Batch, 30, Nodes]

        # Sensor error normalization
        sigma = self.operator.obs_errors.view(1, -1, 1)  # [1, 30, 1]
        diff_normalized = (tb_sim - obs_tb) / sigma      # [Batch, 30, Nodes]

        if obs_mask is not None:
            sq_err = (diff_normalized ** 2) * obs_mask
            denom = torch.sum(obs_mask) + 1e-6
            return torch.sum(sq_err) / denom
        else:
            return torch.mean(diff_normalized ** 2)
