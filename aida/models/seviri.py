#!/usr/bin/env python3
"""
models/seviri.py
----------------
Differentiable SEVIRI Forward Operator H(x) and Radiance Innovation Loss Engine
for AIDA GNN Surrogate Model Training.

Computes simulated brightness temperatures (T_b) for 8 SEVIRI infrared channels
(Channels 4-11: 3.9 um to 13.4 um, including water vapor, CO2, and surface window bands)
across 32 model vertical levels using log-pressure Gaussian weighting functions,
and calculates channel-weighted innovation residuals J_rad_seviri.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# SEVIRI 8-Channel IR Configuration (Channels 4 through 11)
# -----------------------------------------------------------------------------
SEVIRI_CHANNELS = [4, 5, 6, 7, 8, 9, 10, 11]

# Approximate peak pressures (Pa) for SEVIRI channels 4-11:
# IR3.9 (Surface/Low), WV6.2 (Upper-WV), WV7.3 (Mid-WV), IR8.7 (Window),
# IR10.8 (Window), IR12.0 (Window), IR13.4 (CO2 Temperature)
SEVIRI_PEAK_P_PA = np.array([
    100000.0,  # Ch4: IR3.9 (Surface/Low-level)
     35000.0,  # Ch5: WV6.2 (Upper-tropospheric water vapor)
     55000.0,  # Ch6: WV7.3 (Mid-tropospheric water vapor)
    100000.0,  # Ch7: IR8.7 (Surface window / cloud)
    100000.0,  # Ch8: IR10.8 (Surface window / cloud)
    100000.0,  # Ch9: IR12.0 (Surface window)
     95000.0,  # Ch10: IR13.4 (Lower CO2 temperature)
     70000.0   # Ch11: Low/Mid CO2 boundary
], dtype=np.float32)

# Channel Gaussian weighting widths in log-pressure space
SEVIRI_SIGMA_LN_P = np.array([
    0.30, 0.45, 0.45, 0.25, 0.25, 0.25, 0.35, 0.40
], dtype=np.float32)

# Nominal observation errors per channel (Kelvin)
SEVIRI_CHAN_ERRORS = np.array([
    1.50, 1.20, 1.00, 1.10, 1.00, 1.10, 1.30, 1.60
], dtype=np.float32)


class DifferentiableSEVIRIOperator(nn.Module):
    """
    Differentiable Forward Radiance Operator H_seviri(x) in PyTorch.
    Maps atmospheric profile states (Temperature T, Pressure p) to simulated SEVIRI
    brightness temperatures T_b for 8 channels.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(SEVIRI_CHANNELS)

        self.register_buffer("peak_p", torch.from_numpy(SEVIRI_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(SEVIRI_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(SEVIRI_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp_k: Temperature profile tensor [Batch, Levels=32, Nodes] (Kelvin)
            press_pa: Pressure profile tensor [Batch, Levels=32, Nodes] (Pascal)

        Returns:
            Simulated Brightness Temperatures [Batch, Channels=8, Nodes] (Kelvin)
        """
        press_pa = torch.clamp(press_pa, min=10.0)
        ln_press = torch.log(press_pa)  # [B, 32, N]

        ln_peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)  # [1, 8, 1, 1]
        sigma = self.sigma_ln_p.view(1, self.num_channels, 1, 1)

        ln_p_exp = ln_press.unsqueeze(1)  # [B, 1, 32, N]
        temp_exp = temp_k.unsqueeze(1)    # [B, 1, 32, N]

        # Log-pressure Gaussian weighting kernel [B, 8, 32, N]
        weight_logits = -0.5 * ((ln_p_exp - ln_peak_p) / sigma) ** 2
        weights = F.softmax(weight_logits, dim=2)  # Normalize across vertical levels

        tb_sim = torch.sum(temp_exp * weights, dim=2)  # [B, 8, N]
        return tb_sim


class SEVIRIRadianceLoss(nn.Module):
    """Computes normalized SEVIRI satellite radiance innovation loss J_rad_seviri."""
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.operator = DifferentiableSEVIRIOperator(num_levels=num_levels)

    def forward(
        self,
        pred_temp_k: torch.Tensor,
        pred_press_pa: torch.Tensor,
        obs_tb: torch.Tensor,
        obs_mask: torch.Tensor = None
    ) -> torch.Tensor:
        tb_sim = self.operator(pred_temp_k, pred_press_pa)  # [Batch, 8, Nodes]

        sigma = self.operator.obs_errors.view(1, -1, 1)  # [1, 8, 1]
        diff_normalized = (tb_sim - obs_tb) / sigma      # [Batch, 8, Nodes]

        if obs_mask is not None:
            sq_err = (diff_normalized ** 2) * obs_mask
            denom = torch.sum(obs_mask) + 1e-6
            return torch.sum(sq_err) / denom
        else:
            return torch.mean(diff_normalized ** 2)
