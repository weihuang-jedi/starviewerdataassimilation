Here is the implementation to integrate **ATMS (Advanced Technology Microwave Sounder)** 22-channel radiance observations into your AIDA data fetching, model, loss, dataset, and training scripts.

---

### 1. New Module: `models/atms.py`

ATMS combines AMSU-A and MHS heritage channels into a 22-channel microwave suite covering both temperature ($50\text{--}58\text{ GHz}$) and moisture ($183\text{ GHz}$) sounding bands.

```python
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

```

---

### 2. Update `fetch_nnja_amsua_iasi_hms.py` -> `fetch_nnja_obs_suite.py`

Adds NOAA GDAS ATMS BUFR key **`1batms`** (or `atms`) into `SAT_OBS_TYPES`:

```python
# In fetch_nnja_obs_suite.py:

SAT_OBS_TYPES = ["1bamua", "1mtiasi", "1bhms", "1batms"]  # Added 1batms for ATMS

# ATMS Channel Configuration (22 Channels)
ATMS_CHANNELS = np.arange(1, 23, dtype=np.int32)
ATMS_PEAK_HEIGHTS = np.array([
    100.0, 300.0, 700.0, 1500.0, 4000.0, 7000.0, 10000.0, 14000.0,
    18000.0, 22000.0, 26000.0, 30000.0, 35000.0, 40000.0, 45000.0,
    500.0, 1500.0, 3500.0, 6000.0, 9000.0, 12000.0, 16000.0
], dtype=np.float32)

ATMS_CHAN_ERRORS = np.array([
    2.50, 2.20, 1.20, 0.60, 0.30, 0.25, 0.25, 0.25,
    0.25, 0.35, 0.55, 0.80, 1.20, 1.80, 3.50, 2.50,
    2.00, 1.50, 1.20, 1.00, 1.10, 1.30
], dtype=np.float32)

# Inside build_unified_nc_dataset():
    # -------------------------------------------------------------------------
    # 5. Generate ATMS Microwave Radiance Observations
    # -------------------------------------------------------------------------
    lats_atms = np.random.uniform(-80.0, 80.0, n_atms_fovs).astype(np.float32)
    lons_atms = np.random.uniform(-180.0, 180.0, n_atms_fovs).astype(np.float32)
    base_tb_atms = np.array([250, 255, 260, 265, 250, 235, 220, 215, 218, 222, 228, 235, 240, 245, 255, 260, 250, 240, 230, 220, 218, 222], dtype=np.float32)

    for fov in range(n_atms_fovs):
        for ch_idx, ch_num in enumerate(ATMS_CHANNELS):
            tb_val = base_tb_atms[ch_idx] + np.random.normal(0, ATMS_CHAN_ERRORS[ch_idx])
            obs_vars.append("tb")
            obs_vals.append(np.float32(tb_val))
            obs_errs.append(ATMS_CHAN_ERRORS[ch_idx])
            obs_lats.append(lats_atms[fov])
            obs_lons.append(lons_atms[fov])
            obs_z.append(ATMS_PEAK_HEIGHTS[ch_idx])
            obs_channel.append(int(ch_num))
            obs_sensor.append("atms")

```

---

### 3. Update `models/dataset.py`

Update `_load_observations_for_step` to parse `sensor == "atms"`:

```python
# Inside LogStateZarrDataset._load_observations_for_step() in models/dataset.py:

        obs_atms_tb = np.full((22, self.num_nodes), 240.0, dtype=np.float32)
        obs_atms_mask = np.zeros((22, self.num_nodes), dtype=np.float32)

        if obs_file and os.path.exists(obs_file):
            try:
                ds_obs = xr.open_dataset(obs_file)
                vals = ds_obs['observation_value'].values
                sensors = ds_obs['sensor'].values
                channels = ds_obs['channel'].values
                lons = ds_obs['longitude'].values

                # Process ATMS Observations
                mask_atms = (sensors == "atms") & (vals > 100.0) & (vals < 350.0)
                if np.any(mask_atms):
                    ch_atms = channels[mask_atms]
                    val_atms = vals[mask_atms]
                    lon_atms = lons[mask_atms]

                    node_idx_a = ((lon_atms + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx_a = np.clip(node_idx_a, 0, self.num_nodes - 1)

                    for c, v, n in zip(ch_atms, val_atms, node_idx_a):
                        if 1 <= c <= 22:
                            obs_atms_tb[c - 1, n] = v
                            obs_atms_mask[c - 1, n] = 1.0

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
            'obs_atms_tb': torch.from_numpy(obs_atms_tb),
            'obs_atms_mask': torch.from_numpy(obs_atms_mask),
        }

```

---

### 4. Update `models/loss.py` & `models/gnn.py`

Instantiate `DifferentiableATMSOperator` and accept `w_rad_atms` in `AIDASurrogateLoss`:

```python
# In models/loss.py:
from models.atms import DifferentiableATMSOperator

class AIDASurrogateLoss(nn.Module):
    def __init__(
        self,
        w_mse: float = 1.0,
        w_rad_amsua: float = 0.01,
        w_rad_iasi: float = 0.01,
        w_rad_hms: float = 0.01,
        w_rad_atms: float = 0.01,  # Added ATMS loss weight
        num_levels: int = 32,
        **kwargs
    ):
        super().__init__()
        self.w_rad_amsua = w_rad_amsua
        self.w_rad_iasi = w_rad_iasi
        self.w_rad_hms = w_rad_hms
        self.w_rad_atms = w_rad_atms

        self.amsua_loss = DifferentiableAMSUAOperator(num_levels=num_levels)
        self.iasi_loss = DifferentiableIASIOperator(num_levels=num_levels)
        self.hms_loss = DifferentiableHMSOperator(num_levels=num_levels)
        self.atms_loss = DifferentiableATMSOperator(num_levels=num_levels)

```

---

### 5. Update `scripts/train_aida_surrogate.py`

Evaluate ATMS radiance innovation inside `train_epoch()`:

```python
# In scripts/train_aida_surrogate.py:

from models.atms import DifferentiableATMSOperator

# Inside train_epoch():
    obs_atms_tb = batch_data.get('obs_atms_tb', None) if isinstance(batch_data, dict) else None
    obs_atms_mask = batch_data.get('obs_atms_mask', None) if isinstance(batch_data, dict) else None

    w_rad_atms = loss_cfg.get("w_rad_atms", 0.01)
    if obs_atms_tb is not None:
        p_pa = p_hpa.permute(0, 2, 1) * 100.0
        t_k_perm = t_k.permute(0, 2, 1)
        tb_atms_sim = atms_op(t_k_perm, p_pa)  # [B, 22, N]

        tb_atms_obs = obs_atms_tb.to(device)
        if tb_atms_obs.shape[1] != 22 and tb_atms_obs.shape[2] == 22:
            tb_atms_obs = tb_atms_obs.permute(0, 2, 1)

        if tb_atms_sim.shape[1] != 22 and tb_atms_sim.shape[2] == 22:
            tb_atms_sim = tb_atms_sim.permute(0, 2, 1)

        err_atms = atms_obs_err.view(1, 22, 1)
        innov_atms = (tb_atms_obs - tb_atms_sim) / err_atms

        if obs_atms_mask is not None:
            mask_atms = obs_atms_mask.to(device)
            if mask_atms.shape[1] != 22 and mask_atms.shape[2] == 22:
                mask_atms = mask_atms.permute(0, 2, 1)
            loss_rad_atms = torch.sum((innov_atms ** 2) * mask_atms) / (22.0 * torch.sum(mask_atms) + 1e-8)
        else:
            loss_rad_atms = torch.mean(innov_atms ** 2) / 22.0
    else:
        loss_rad_atms = torch.tensor(0.0, device=device)

    total_loss += (w_rad_atms * loss_rad_atms)
    metrics["loss_rad_atms"] = loss_rad_atms.item()

# Inside train_model():
    atms_op = DifferentiableATMSOperator(num_levels=mesh_cfg.get("num_levels", 32)).to(device)
    atms_obs_err = atms_op.obs_errors.to(device)
    print(f"[TRAIN] ATMS Radiance Weight: {loss_cfg.get('w_rad_atms', 0.01)}", flush=True)

```

---

### 6. Update `models/__init__.py`

```python
# In models/__init__.py:

from .dataset import LogStateZarrDataset, SyntheticAIDAStateDataset
from .gnn import IcosahedralGNNSurrogate
from .loss import AIDASurrogateLoss
from .amsua import DifferentiableAMSUAOperator
from .iasi import DifferentiableIASIOperator
from .hms import DifferentiableHMSOperator
from .atms import DifferentiableATMSOperator

__all__ = [
    "LogStateZarrDataset",
    "SyntheticAIDAStateDataset",
    "IcosahedralGNNSurrogate",
    "AIDASurrogateLoss",
    "DifferentiableAMSUAOperator",
    "DifferentiableIASIOperator",
    "DifferentiableHMSOperator",
    "DifferentiableATMSOperator",
]

```
