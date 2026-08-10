Here is the complete implementation to integrate the **Hyperspectral / High-resolution Microwave Sounder (HMS)** 12-channel radiance loss into your AIDA pipeline.

---

### 1. New Module: `models/hms.py`

This module defines `DifferentiableHMSOperator` and `HMSRadianceLoss` for the 12 HMS channels ($50\text{--}58\text{ GHz}$ temperature and $183\text{ GHz}$ humidity sounding bands).

```python
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

```

---

### 2. Update `models/gnn.py`

Add `DifferentiableHMSOperator` instantiation and exposed evaluation call in `models/gnn.py`:

```python
# Insert imports at top of models/gnn.py:
from models.amsua import DifferentiableAMSUAOperator
from models.iasi import DifferentiableIASIOperator
from models.hms import DifferentiableHMSOperator


class IcosahedralGNNSurrogate(nn.Module):
    def __init__(
        self,
        in_vars: int = 7,
        hidden_dim: int = 64,
        num_levels: int = 32,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_vars = in_vars
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_layers = num_layers

        # ... (GNN backbone layers setup) ...

        # Instantiate Differentiable Radiance Operators
        self.amsua_operator = DifferentiableAMSUAOperator(num_levels=num_levels)
        self.iasi_operator = DifferentiableIASIOperator(num_levels=num_levels)
        self.hms_operator = DifferentiableHMSOperator(num_levels=num_levels)

    def forward_hms_radiances(self, temp_k: torch.Tensor, press_pa: torch.Tensor) -> torch.Tensor:
        """Exposes direct forward evaluation of HMS brightness temperatures."""
        return self.hms_operator(temp_k, press_pa)

```

---

### 3. Update `models/dataset.py`

Update `LogStateZarrDataset` and `SyntheticAIDAStateDataset` to extract HMS observation arrays (`obs_hms_tb`, `obs_hms_mask`) from the unified NetCDF files:

```python
# Inside LogStateZarrDataset._load_observations_for_step(self, idx) in models/dataset.py:

        # Allocate placeholder array for 12 HMS channels across nodes
        obs_hms_tb = np.full((12, self.num_nodes), 240.0, dtype=np.float32)
        obs_hms_mask = np.zeros((12, self.num_nodes), dtype=np.float32)

        if obs_file and os.path.exists(obs_file):
            try:
                ds_obs = xr.open_dataset(obs_file)
                vals = ds_obs['observation_value'].values
                sensors = ds_obs['sensor'].values
                channels = ds_obs['channel'].values
                lats = ds_obs['latitude'].values
                lons = ds_obs['longitude'].values

                # Process HMS Observations
                mask_hms = (sensors == "hms") & (vals > 100.0) & (vals < 350.0)
                if np.any(mask_hms):
                    ch_hms = channels[mask_hms]
                    val_hms = vals[mask_hms]
                    lon_hms = lons[mask_hms]

                    node_idx_h = ((lon_hms + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx_h = np.clip(node_idx_h, 0, self.num_nodes - 1)

                    for c, v, n in zip(ch_hms, val_hms, node_idx_h):
                        if 1 <= c <= 12:
                            obs_hms_tb[c - 1, n] = v
                            obs_hms_mask[c - 1, n] = 1.0

                ds_obs.close()
            except Exception:
                pass

        return {
            'obs_amsua_tb': torch.from_numpy(obs_amsua_tb),
            'obs_amsua_mask': torch.from_numpy(obs_amsua_mask),
            'obs_iasi_tb': torch.from_numpy(obs_iasi_tb),
            'obs_iasi_mask': torch.from_numpy(obs_iasi_mask),
            'obs_hms_tb': torch.from_numpy(obs_hms_tb),
            'obs_hms_mask': torch.from_numpy(obs_hms_mask),
        }

```

And update `SyntheticAIDAStateDataset.__getitem__`:

```python
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            'background': torch.from_numpy(self.data_x[idx]),
            'target': torch.from_numpy(self.data_y[idx]),
            'obs_amsua_tb': torch.full((15, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_amsua_mask': torch.ones((15, self.num_nodes), dtype=torch.float32),
            'obs_iasi_tb': torch.full((30, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_iasi_mask': torch.ones((30, self.num_nodes), dtype=torch.float32),
            'obs_hms_tb': torch.full((12, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_hms_mask': torch.ones((12, self.num_nodes), dtype=torch.float32),
        }

```

---

### 4. Update `models/loss.py`

Add `w_rad_hms` weight parameter and `DifferentiableHMSOperator` inside `AIDASurrogateLoss`:

```python
# In models/loss.py

from models.amsua import DifferentiableAMSUAOperator
from models.iasi import DifferentiableIASIOperator
from models.hms import DifferentiableHMSOperator


class AIDASurrogateLoss(nn.Module):
    def __init__(
        self,
        w_mse: float = 1.0,
        w_rad_amsua: float = 0.01,
        w_rad_iasi: float = 0.01,
        w_rad_hms: float = 0.01,     # Added HMS weight key
        num_levels: int = 32,
        **kwargs
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_rad_amsua = w_rad_amsua
        self.w_rad_iasi = w_rad_iasi
        self.w_rad_hms = w_rad_hms

        self.amsua_loss = DifferentiableAMSUAOperator(num_levels=num_levels)
        self.iasi_loss = DifferentiableIASIOperator(num_levels=num_levels)
        self.hms_loss = DifferentiableHMSOperator(num_levels=num_levels)

```

---

### 5. Update `scripts/train_aida_surrogate.py`

Update `train_epoch()` and `train_model()` to evaluate HMS radiance innovations during training:

```python
# In scripts/train_aida_surrogate.py:

# 1. Update imports
from models.hms import DifferentiableHMSOperator

# 2. Inside train_epoch():
def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    edge_index,
    graph_mesh_ops,
    amsua_op,
    amsua_obs_err,
    iasi_op,
    iasi_obs_err,
    hms_op,
    hms_obs_err,
    loss_cfg
):
    # Unpack HMS batch tensors
    obs_hms_tb = batch_data.get('obs_hms_tb', None) if isinstance(batch_data, dict) else None
    obs_hms_mask = batch_data.get('obs_hms_mask', None) if isinstance(batch_data, dict) else None

    # ... (Base loss & un-normalization p_hpa, t_k) ...

    # -------------------------------------------------------------------------
    # 5. HMS Satellite Radiance Innovation Loss
    # -------------------------------------------------------------------------
    w_rad_hms = loss_cfg.get("w_rad_hms", 0.01)
    if obs_hms_tb is not None:
        p_pa = p_hpa.permute(0, 2, 1) * 100.0
        t_k_perm = t_k.permute(0, 2, 1)
        tb_hms_sim = hms_op(t_k_perm, p_pa)  # [B, 12, N]

        tb_hms_obs = obs_hms_tb.to(device)
        if tb_hms_obs.shape[1] != 12 and tb_hms_obs.shape[2] == 12:
            tb_hms_obs = tb_hms_obs.permute(0, 2, 1)

        if tb_hms_sim.shape[1] != 12 and tb_hms_sim.shape[2] == 12:
            tb_hms_sim = tb_hms_sim.permute(0, 2, 1)

        err_hms = hms_obs_err.view(1, 12, 1)
        innov_hms = (tb_hms_obs - tb_hms_sim) / err_hms

        if obs_hms_mask is not None:
            mask_hms = obs_hms_mask.to(device)
            if mask_hms.shape[1] != 12 and mask_hms.shape[2] == 12:
                mask_hms = mask_hms.permute(0, 2, 1)
            loss_rad_hms = torch.sum((innov_hms ** 2) * mask_hms) / (12.0 * torch.sum(mask_hms) + 1e-8)
        else:
            loss_rad_hms = torch.mean(innov_hms ** 2) / 12.0
    else:
        loss_rad_hms = torch.tensor(0.0, device=device)

    # Add J_rad_hms to total loss objective
    total_loss = loss + (w_conv * loss_conv) + (w_rad_amsua * loss_rad_amsua) + (w_rad_iasi * loss_rad_iasi) + (w_rad_hms * loss_rad_hms)
    metrics["loss_rad_hms"] = loss_rad_hms.item()


# 3. Inside train_model():
    # Instantiate HMS Forward Operator & Observation Error Buffers
    hms_op = DifferentiableHMSOperator(num_levels=mesh_cfg.get("num_levels", 32)).to(device)
    hms_obs_err = hms_op.obs_errors.to(device)

    print(f"[TRAIN] HMS Radiance Weight: {loss_cfg.get('w_rad_hms', 0.01)}", flush=True)

```

---

### 6. Update `models/__init__.py`

Ensure `DifferentiableHMSOperator` is exposed in `models/__init__.py`:

```python
from .dataset import LogStateZarrDataset, SyntheticAIDAStateDataset
from .gnn import IcosahedralGNNSurrogate
from .loss import AIDASurrogateLoss
from .amsua import DifferentiableAMSUAOperator
from .iasi import DifferentiableIASIOperator
from .hms import DifferentiableHMSOperator

__all__ = [
    "LogStateZarrDataset",
    "SyntheticAIDAStateDataset",
    "IcosahedralGNNSurrogate",
    "AIDASurrogateLoss",
    "DifferentiableAMSUAOperator",
    "DifferentiableIASIOperator",
    "DifferentiableHMSOperator",
]

```
