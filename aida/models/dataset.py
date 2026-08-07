#!/usr/bin/env python3
"""
models/dataset.py
-----------------
Dataset wrappers for multi-variable Zarr stores and unified satellite/conventional
observations (AMSU-A + IASI) for AIDA AI-DA surrogate training.
"""

import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

LOG_STATE_VARS = [
    'ln_t_icosahedral',
    'u_icosahedral',
    'v_icosahedral',
    'w_icosahedral',
    'q_icosahedral',
    'ln_rho_icosahedral',
    'ln_p_icosahedral'
]


class LogStateZarrDataset(Dataset):
    """
    Multi-Variable Zarr Dataset with Observation Ingestion (AMSU-A + IASI).
    
    Reads background state pairs (x_t, x_t+1) and aligns time-corresponding
    satellite observation files from `obs_dir` (e.g., `conv_amsua_iasi_2024/`).
    """
    def __init__(self, zarr_path: str, obs_dir: str = "conv_amsua_iasi_2024"):
        super().__init__()
        self.zarr_path = zarr_path
        self.obs_dir = obs_dir
        self.var_keys = LOG_STATE_VARS

        try:
            import zarr
        except ImportError:
            raise ImportError("zarr library is required. Run 'pip install zarr'.")

        self.root = zarr.open(zarr_path, mode='r')

        available_keys = list(self.root.array_keys())
        for k in self.var_keys:
            if k not in available_keys:
                raise KeyError(
                    f"Expected key '{k}' not found in Zarr store at '{zarr_path}'. "
                    f"Found: {available_keys}"
                )

        first_arr = self.root[self.var_keys[0]]
        self.num_time_steps = first_arr.shape[0] - 1  # t -> t+1 pairs
        self.num_vars = len(self.var_keys)
        self.num_levels = first_arr.shape[1]  # 32
        self.num_nodes = first_arr.shape[2]   # 2562

        # Extract timestamps if present in Zarr metadata
        if 'time' in self.root:
            self.timestamps = list(self.root['time'][:])
        else:
            self.timestamps = None

        print(f"[DATASET] Loaded Multi-Array Zarr dataset from '{zarr_path}'")
        print(f"          Variables ({self.num_vars}): {self.var_keys}")
        print(
            f"          Dimensions: Time={self.num_time_steps + 1}, "
            f"Levels={self.num_levels}, Nodes={self.num_nodes}"
        )
        print(f"          Observation Directory: '{obs_dir}'")

    def __len__(self):
        return self.num_time_steps

    def _load_observations_for_step(self, idx: int) -> dict:
        """
        Locates and loads satellite brightness temperatures (AMSU-A + IASI)
        for the given time index `idx` from NetCDF observation files.
        """
        # Default placeholder arrays [15, 2562] and [30, 2562]
        obs_amsua_tb = np.full((15, self.num_nodes), 240.0, dtype=np.float32)
        obs_amsua_mask = np.zeros((15, self.num_nodes), dtype=np.float32)

        obs_iasi_tb = np.full((30, self.num_nodes), 240.0, dtype=np.float32)
        obs_iasi_mask = np.zeros((30, self.num_nodes), dtype=np.float32)

        if not os.path.exists(self.obs_dir):
            return {
                'obs_amsua_tb': torch.from_numpy(obs_amsua_tb),
                'obs_amsua_mask': torch.from_numpy(obs_amsua_mask),
                'obs_iasi_tb': torch.from_numpy(obs_iasi_tb),
                'obs_iasi_mask': torch.from_numpy(obs_iasi_mask),
            }

        # Match observation NetCDF files in obs_dir
        obs_files = sorted(glob.glob(os.path.join(self.obs_dir, "obs_unified.*.nc")))
        if not obs_files or idx >= len(obs_files):
            # Fallback if specific timestep index exceeds available files
            obs_file = obs_files[idx % len(obs_files)] if obs_files else None
        else:
            obs_file = obs_files[idx]

        if obs_file and os.path.exists(obs_file):
            try:
                ds_obs = xr.open_dataset(obs_file)

                # Read observation values, variables, sensors, channels, and coordinates
                vals = ds_obs['observation_value'].values
                sensors = ds_obs['sensor'].values
                channels = ds_obs['channel'].values
                lats = ds_obs['latitude'].values
                lons = ds_obs['longitude'].values

                # Process AMSU-A Observations
                mask_amsua = (sensors == "amsua") & (vals > 100.0) & (vals < 350.0)
                if np.any(mask_amsua):
                    ch_amsua = channels[mask_amsua]
                    val_amsua = vals[mask_amsua]
                    lat_amsua = lats[mask_amsua]
                    lon_amsua = lons[mask_amsua]

                    # Map coordinates to node indices [0..2561]
                    node_idx = ((lon_amsua + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx = np.clip(node_idx, 0, self.num_nodes - 1)

                    for c, v, n in zip(ch_amsua, val_amsua, node_idx):
                        if 1 <= c <= 15:
                            obs_amsua_tb[c - 1, n] = v
                            obs_amsua_mask[c - 1, n] = 1.0

                # Process IASI Observations
                mask_iasi = (sensors == "iasi") & (vals > 100.0) & (vals < 350.0)
                if np.any(mask_iasi):
                    ch_iasi = channels[mask_iasi]
                    val_iasi = vals[mask_iasi]
                    lat_iasi = lats[mask_iasi]
                    lon_iasi = lons[mask_iasi]

                    node_idx_i = ((lon_iasi + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx_i = np.clip(node_idx_i, 0, self.num_nodes - 1)

                    # Map IASI channels (1..30)
                    for c, v, n in zip(ch_iasi, val_iasi, node_idx_i):
                        c_idx = c - 1 if c <= 30 else 0
                        if 0 <= c_idx < 30:
                            obs_iasi_tb[c_idx, n] = v
                            obs_iasi_mask[c_idx, n] = 1.0

                ds_obs.close()
            except Exception:
                pass

        return {
            'obs_amsua_tb': torch.from_numpy(obs_amsua_tb),
            'obs_amsua_mask': torch.from_numpy(obs_amsua_mask),
            'obs_iasi_tb': torch.from_numpy(obs_iasi_tb),
            'obs_iasi_mask': torch.from_numpy(obs_iasi_mask),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x_list = [np.array(self.root[key][idx], dtype=np.float32) for key in self.var_keys]
        y_list = [np.array(self.root[key][idx + 1], dtype=np.float32) for key in self.var_keys]

        x = np.stack(x_list, axis=0)
        y = np.stack(y_list, axis=0)

        # Basic NaN safeguard on data read
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        item_dict = {
            'background': torch.from_numpy(x),
            'target': torch.from_numpy(y)
        }

        # Ingest time-corresponding satellite observation fields
        obs_dict = self._load_observations_for_step(idx)
        item_dict.update(obs_dict)

        return item_dict


class SyntheticAIDAStateDataset(Dataset):
    """Fallback dataset simulating log-state atmospheric variables on mesh with dummy observations."""
    def __init__(self, num_samples: int = 80, num_nodes: int = 2562, num_levels: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_levels = num_levels
        self.num_vars = len(LOG_STATE_VARS)

        np.random.seed(42)
        self.data_x = np.random.randn(num_samples, 7, num_levels, num_nodes).astype(np.float32)
        self.data_y = self.data_x * 0.98 + 0.02 * np.random.randn(num_samples, 7, num_levels, num_nodes).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            'background': torch.from_numpy(self.data_x[idx]),
            'target': torch.from_numpy(self.data_y[idx]),
            'obs_amsua_tb': torch.full((15, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_amsua_mask': torch.ones((15, self.num_nodes), dtype=torch.float32),
            'obs_iasi_tb': torch.full((30, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_iasi_mask': torch.ones((30, self.num_nodes), dtype=torch.float32),
        }
