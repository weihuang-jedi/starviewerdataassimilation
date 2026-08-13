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
        # Allocate placeholder array for 12 HMS channels across nodes
        obs_hms_tb = np.full((12, self.num_nodes), 240.0, dtype=np.float32)
        obs_hms_mask = np.zeros((12, self.num_nodes), dtype=np.float32)

        # Default placeholder arrays [15, 2562] and [30, 2562]
        obs_amsua_tb = np.full((15, self.num_nodes), 240.0, dtype=np.float32)
        obs_amsua_mask = np.zeros((15, self.num_nodes), dtype=np.float32)

        obs_iasi_tb = np.full((30, self.num_nodes), 240.0, dtype=np.float32)
        obs_iasi_mask = np.zeros((30, self.num_nodes), dtype=np.float32)

        obs_atms_tb = np.full((22, self.num_nodes), 240.0, dtype=np.float32)
        obs_atms_mask = np.zeros((22, self.num_nodes), dtype=np.float32)

        obs_cris_tb = np.full((30, self.num_nodes), 240.0, dtype=np.float32)
        obs_cris_mask = np.zeros((30, self.num_nodes), dtype=np.float32)

        obs_seviri_tb = np.full((8, self.num_nodes), 240.0, dtype=np.float32)
        obs_seviri_mask = np.zeros((8, self.num_nodes), dtype=np.float32)

        obs_gsrasr_tb = np.full((10, self.num_nodes), 240.0, dtype=np.float32)
        obs_gsrasr_mask = np.zeros((10, self.num_nodes), dtype=np.float32)

        obs_gsrcsr_tb = np.full((7, self.num_nodes), 240.0, dtype=np.float32)
        obs_gsrcsr_mask = np.zeros((7, self.num_nodes), dtype=np.float32)

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

                # Process CRIS Observations
                mask_cris = (sensors == "cris") & (vals > 100.0) & (vals < 350.0)
                if np.any(mask_cris):
                    ch_cris = channels[mask_cris]
                    val_cris = vals[mask_cris]
                    lon_cris = lons[mask_cris]

                    node_idx_c = ((lon_cris + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx_c = np.clip(node_idx_c, 0, self.num_nodes - 1)

                    for c, v, n in zip(ch_cris, val_cris, node_idx_c):
                        c_idx = c - 1 if c <= 30 else 0
                        if 0 <= c_idx < 30:
                            obs_cris_tb[c_idx, n] = v
                            obs_cris_mask[c_idx, n] = 1.0

                # Process SEVIRI Observations
                mask_seviri = (sensors == "seviri") & (vals > 100.0) & (vals < 350.0)
                if np.any(mask_seviri):
                    ch_seviri = channels[mask_seviri]
                    val_seviri = vals[mask_seviri]
                    lon_seviri = lons[mask_seviri]

                    node_idx_s = ((lon_seviri + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx_s = np.clip(node_idx_s, 0, self.num_nodes - 1)

                    # SEVIRI uses channels 4 through 11 -> mapped to 0..7
                    for c, v, n in zip(ch_seviri, val_seviri, node_idx_s):
                        c_idx = c - 4
                        if 0 <= c_idx < 8:
                            obs_seviri_tb[c_idx, n] = v
                            obs_seviri_mask[c_idx, n] = 1.0

                # Process GSRASR Observations
                mask_gsrasr = (sensors == "gsrasr") & (vals > 100.0) & (vals < 350.0)
                if np.any(mask_gsrasr):
                    ch_gsrasr = channels[mask_gsrasr]
                    val_gsrasr = vals[mask_gsrasr]
                    lon_gsrasr = lons[mask_gsrasr]

                    node_idx_g = ((lon_gsrasr + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx_g = np.clip(node_idx_g, 0, self.num_nodes - 1)

                    # GSRASR uses channels 7 through 16 -> mapped to 0..9
                    for c, v, n in zip(ch_gsrasr, val_gsrasr, node_idx_g):
                        c_idx = c - 7
                        if 0 <= c_idx < 10:
                            obs_gsrasr_tb[c_idx, n] = v
                            obs_gsrasr_mask[c_idx, n] = 1.0

                # Process GSRCSR Observations
                mask_gsrcsr = (sensors == "gsrcsr") & (vals > 100.0) & (vals < 350.0)
                if np.any(mask_gsrcsr):
                    ch_gsrcsr = channels[mask_gsrcsr]
                    val_gsrcsr = vals[mask_gsrcsr]
                    lon_gsrcsr = lons[mask_gsrcsr]

                    node_idx_gc = ((lon_gsrcsr + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx_gc = np.clip(node_idx_gc, 0, self.num_nodes - 1)

                    # Channel mapping for GSRCSR [8, 9, 10, 12, 13, 14, 15] -> 0..6
                    channel_map = {8: 0, 9: 1, 10: 2, 12: 3, 13: 4, 14: 5, 15: 6}
                    for c, v, n in zip(ch_gsrcsr, val_gsrcsr, node_idx_gc):
                        if c in channel_map:
                            c_idx = channel_map[c]
                            obs_gsrcsr_tb[c_idx, n] = v
                            obs_gsrcsr_mask[c_idx, n] = 1.0

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
            'obs_cris_tb': torch.from_numpy(obs_cris_tb),
            'obs_cris_mask': torch.from_numpy(obs_cris_mask),
            'obs_seviri_tb': torch.from_numpy(obs_seviri_tb),
            'obs_seviri_mask': torch.from_numpy(obs_seviri_mask),
            'obs_gsrasr_tb': torch.from_numpy(obs_gsrasr_tb),
            'obs_gsrasr_mask': torch.from_numpy(obs_gsrasr_mask),
            'obs_gsrcsr_tb': torch.from_numpy(obs_gsrcsr_tb),
            'obs_gsrcsr_mask': torch.from_numpy(obs_gsrcsr_mask),
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
            'obs_hms_tb': torch.full((12, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_hms_mask': torch.ones((12, self.num_nodes), dtype=torch.float32),
            'obs_atms_tb': torch.full((12, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_atms_mask': torch.ones((12, self.num_nodes), dtype=torch.float32),
            'obs_cris_tb': torch.full((12, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_cris_mask': torch.ones((12, self.num_nodes), dtype=torch.float32),
            'obs_seviri_tb': torch.full((8, self.num_nodes), 240.0, dtype=torch.torch.float32),
            'obs_seviri_mask': torch.ones((8, self.num_nodes), dtype=torch.float32),
            'obs_gsrasr_tb': torch.full((10, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_gsrasr_mask': torch.ones((10, self.num_nodes), dtype=torch.float32),
            'obs_gsrcsr_tb': torch.full((7, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_gsrcsr_mask': torch.ones((7, self.num_nodes), dtype=torch.float32),
        }
