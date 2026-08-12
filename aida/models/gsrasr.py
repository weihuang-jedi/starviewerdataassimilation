#!/usr/bin/env python3
"""
models/gsrasr.py
----------------
Differentiable GOES All-Sky Radiance (GSRASR / ABI) Forward Operator H(x)
and Radiance Innovation Loss Engine for AIDA GNN Surrogate Model Training.

Computes simulated brightness temperatures (T_b) for 10 GOES ABI IR channels
(Channels 7-16: 3.9 um to 13.3 um, including upper, mid, and lower-level water vapor,
ozone, surface window, and CO2 temperature bands) across 32 model vertical levels
using log-pressure Gaussian weighting functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# GOES ABI 10-Channel IR Sounding Subset (Channels 7 through 16)
# -----------------------------------------------------------------------------
GSRASR_CHANNELS = list(range(7, 17))

# Approximate peak pressures (Pa) for GOES ABI channels 7-16:
# Ch7: 3.9um (Low/Surf), Ch8: 6.2um (Upper WV), Ch9: 6.9um (Mid WV), Ch10: 7.3um (Lower WV),
# Ch11: 8.4um (Window), Ch12: 9.6um (Ozone), Ch13: 10.3um (Clean IR), Ch14: 11.2um (Window),
# Ch15: 12.3um (Dirty IR), Ch16: 13.3um (CO2 Temp)
GSRASR_PEAK_P_PA = np.array([
    100000.0,  # Ch7:  3.9 um - Surface / Low-level
     32000.0,  # Ch8:  6.2 um - Upper-tropospheric Water Vapor
     48000.0,  # Ch9:  6.9 um - Mid-tropospheric Water Vapor
     62000.0,  # Ch10: 7.3 um - Lower-tropospheric Water Vapor
    100000.0,  # Ch11: 8.4 um - Surface / Cloud Top
     25000.0,  # Ch12: 9.6 um - Stratospheric Ozone
    100000.0,  # Ch13: 10.3 um - Clean Surface IR Window
    100000.0,  # Ch14: 11.2 um - Longwave Surface Window
     95000.0,  # Ch15: 12.3 um - Lower Atmospheric Moisture
     80000.0   # Ch16: 13.3 um - CO2 Temperature Sounding
], dtype=np.float32)

# Channel Gaussian weighting widths in log-pressure space
GSRASR_SIGMA_LN_P = np.array([
    0.30, 0.45, 0.45, 0.40, 0.25, 0.35, 0.25, 0.25, 0.30, 0.35
], dtype=np.float32)

# Nominal observation errors per channel (Kelvin)
GSRASR_CHAN_ERRORS = np.array([
    1.40, 1.15, 1.05, 0.95, 1.10, 1.25, 1.00, 1.00, 1.15, 1.35
], dtype=np.float32)


class DifferentiableGSRASROperator(nn.Module):
    """
    Differentiable Forward Radiance Operator H_gsrasr(x) in PyTorch.
    Maps atmospheric profile states (Temperature T, Pressure p) to simulated GOES ABI
    brightness temperatures T_b for 10 channels.
    """
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(GSRASR_CHANNELS)

        self.register_buffer("peak_p", torch.from_numpy(GSRASR_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(GSRASR_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(GSRASR_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temp_k: Temperature profile tensor [Batch, Levels=32, Nodes] (Kelvin)
            press_pa: Pressure profile tensor [Batch, Levels=32, Nodes] (Pascal)

        Returns:
            Simulated Brightness Temperatures [Batch, Channels=10, Nodes] (Kelvin)
        """
        press_pa = torch.clamp(press_pa, min=10.0)
        ln_press = torch.log(press_pa)  # [B, 32, N]

        ln_peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)  # [1, 10, 1, 1]
        sigma = self.sigma_ln_p.view(1, self.num_channels, 1, 1)

        ln_p_exp = ln_press.unsqueeze(1)  # [B, 1, 32, N]
        temp_exp = temp_k.unsqueeze(1)    # [B, 1, 32, N]

        # Log-pressure Gaussian weighting kernel [B, 10, 32, N]
        weight_logits = -0.5 * ((ln_p_exp - ln_peak_p) / sigma) ** 2
        weights = F.softmax(weight_logits, dim=2)

        tb_sim = torch.sum(temp_exp * weights, dim=2)  # [B, 10, N]
        return tb_sim


class GSRASRRadianceLoss(nn.Module):
    """Computes normalized GOES All-Sky Radiance innovation loss J_rad_gsrasr."""
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.operator = DifferentiableGSRASROperator(num_levels=num_levels)

    def forward(
        self,
        pred_temp_k: torch.Tensor,
        pred_press_pa: torch.Tensor,
        obs_tb: torch.Tensor,
        obs_mask: torch.Tensor = None
    ) -> torch.Tensor:
        tb_sim = self.operator(pred_temp_k, pred_press_pa)  # [Batch, 10, Nodes]

        sigma = self.operator.obs_errors.view(1, -1, 1)  # [1, 10, 1]
        diff_normalized = (tb_sim - obs_tb) / sigma      # [Batch, 10, Nodes]

        if obs_mask is not None:
            sq_err = (diff_normalized ** 2) * obs_mask
            denom = torch.sum(obs_mask) + 1e-6
            return torch.sum(sq_err) / denom
        else:
            return torch.mean(diff_normalized ** 2)
