#!/usr/bin/env python3
"""
models/gsrcsr.py
----------------
Differentiable GOES Clear-Sky Radiance (GSRCSR / ABI CSR) Forward Operator H(x)
and Radiance Innovation Loss Engine for AIDA GNN Surrogate Model Training.

Computes simulated brightness temperatures (T_b) for 7 GOES Clear-Sky IR channels
(Channels 8-10: Water Vapor, Channels 12-15: Window/Moisture) across 32 model vertical
levels using log-pressure Gaussian weighting functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# GOES Clear-Sky Radiance 7-Channel Sounder Subset
# -----------------------------------------------------------------------------
GSRCSR_CHANNELS = [8, 9, 10, 12, 13, 14, 15]

# Approximate peak pressures (Pa) for GOES CSR channels:
# Ch8: 6.2um (Upper WV), Ch9: 6.9um (Mid WV), Ch10: 7.3um (Lower WV),
# Ch12: 9.6um (Ozone), Ch13: 10.3um (Clean IR), Ch14: 11.2um (Window), Ch15: 12.3um (Dirty IR)
GSRCSR_PEAK_P_PA = np.array([
     32000.0,  # Ch8:  6.2 um - Upper-tropospheric Water Vapor
     48000.0,  # Ch9:  6.9 um - Mid-tropospheric Water Vapor
     62000.0,  # Ch10: 7.3 um - Lower-tropospheric Water Vapor
     25000.0,  # Ch12: 9.6 um - Stratospheric Ozone
    100000.0,  # Ch13: 10.3 um - Clean Surface IR Window
    100000.0,  # Ch14: 11.2 um - Longwave Surface Window
     95000.0   # Ch15: 12.3 um - Lower Atmospheric Moisture
], dtype=np.float32)

# Channel Gaussian weighting widths in log-pressure space
GSRCSR_SIGMA_LN_P = np.array([
    0.45, 0.45, 0.40, 0.35, 0.25, 0.25, 0.30
], dtype=np.float32)

# Nominal observation errors per channel (Kelvin)
GSRCSR_CHAN_ERRORS = np.array([
    1.10, 1.00, 0.90, 1.20, 0.95, 0.95, 1.10
], dtype=np.float32)


class DifferentiableGSRCSR0perator(nn.Module):
    """
    Differentiable Forward Radiance Operator H_gsrcsr(x) in PyTorch.
    Maps atmospheric profile states (Temperature T, Pressure p) to simulated GOES CSR
    brightness temperatures T_b for 7 channels.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(GSRCSR_CHANNELS)

        self.register_buffer("peak_p", torch.from_numpy(GSRCSR_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(GSRCSR_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(GSRCSR_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp_k: Temperature profile tensor [Batch, Levels=32, Nodes] (Kelvin)
            press_pa: Pressure profile tensor [Batch, Levels=32, Nodes] (Pascal)

        Returns:
            Simulated Brightness Temperatures [Batch, Channels=7, Nodes] (Kelvin)
        """
        press_pa = torch.clamp(press_pa, min=10.0)
        ln_press = torch.log(press_pa)  # [B, 32, N]

        ln_peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)  # [1, 7, 1, 1]
        sigma = self.sigma_ln_p.view(1, self.num_channels, 1, 1)

        ln_p_exp = ln_press.unsqueeze(1)  # [B, 1, 32, N]
        temp_exp = temp_k.unsqueeze(1)    # [B, 1, 32, N]

        # Log-pressure Gaussian weighting kernel [B, 7, 32, N]
        weight_logits = -0.5 * ((ln_p_exp - ln_peak_p) / sigma) ** 2
        weights = F.softmax(weight_logits, dim=2)

        tb_sim = torch.sum(temp_exp * weights, dim=2)  # [B, 7, N]
        return tb_sim


class GSRCSRRadianceLoss(nn.Module):
    """Computes normalized GOES Clear-Sky Radiance innovation loss J_rad_gsrcsr."""
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.operator = DifferentiableGSRCSR0perator(num_levels=num_levels)

    def forward(
        self,
        pred_temp_k: torch.Tensor,
        pred_press_pa: torch.Tensor,
        obs_tb: torch.Tensor,
        obs_mask: torch.Tensor = None
    ) -> torch.Tensor:
        tb_sim = self.operator(pred_temp_k, pred_press_pa)  # [Batch, 7, Nodes]

        sigma = self.operator.obs_errors.view(1, -1, 1)  # [1, 7, 1]
        diff_normalized = (tb_sim - obs_tb) / sigma      # [Batch, 7, Nodes]

        if obs_mask is not None:
            sq_err = (diff_normalized ** 2) * obs_mask
            denom = torch.sum(obs_mask) + 1e-6
            return torch.sum(sq_err) / denom
        else:
            return torch.mean(diff_normalized ** 2)
