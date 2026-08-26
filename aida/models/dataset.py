#!/usr/bin/env python3
"""
models/dataset.py
-----------------
Dataset Loaders for AIDA GNN Surrogate Model Training & AI Data Assimilation.
Extracts 3D dynamic atmospheric log-state fields, 3D terrain-following geometric
height profiles (h_3d), 2D static topography features (static_topo), dynamic
Solar Zenith Angle cos(SZA) forcing, and Surface Roughness z0 from Zarr datasets
alongside multi-sensor satellite and conventional observations.
Includes vertical Pressure-to-Height (p -> z) interpolation for conventional observations.
Target is Ground Truth Analysis A_0h at time 0h.
"""

import os
import glob
import re
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from datetime import datetime


def compute_solar_zenith_angle(lats_deg: np.ndarray, lons_deg: np.ndarray, timestamp_unix: float) -> np.ndarray:
    """Computes cosine of Solar Zenith Angle cos(SZA) in [0.0, 1.0] across mesh nodes."""
    dt = datetime.utcfromtimestamp(timestamp_unix)
    day_of_year = dt.timetuple().tm_yday
    hour_utc = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    rad = np.pi / 180.0
    declination = 23.45 * np.sin(rad * (360.0 / 365.0) * (day_of_year - 81)) * rad
    solar_time = hour_utc + (lons_deg / 15.0)
    hour_angle = (solar_time - 12.0) * 15.0 * rad

    lats_rad = lats_deg * rad
    cos_sza = np.sin(lats_rad) * np.sin(declination) + np.cos(lats_rad) * np.cos(declination) * np.cos(hour_angle)
    return np.maximum(0.0, cos_sza).astype(np.float32)


class LogStateZarrDataset(Dataset):
    """
    Standard Zarr Dataset Loader for Single-Step AI-DA State Ingestion.
    Loads background B_0h and target analysis A_0h along with 3D terrain heights (h_3d),
    static surface topography features, surface roughness z0, and solar zenith angle cos(SZA).
    """
    def __init__(self, zarr_path: str, obs_dir: str = None):
        super().__init__()
        self.zarr_path = zarr_path
        self.obs_dir = obs_dir

        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"[ERROR] Zarr dataset not found at '{zarr_path}'")

        print(f"[DATASET] Loading Zarr dataset from: '{zarr_path}'", flush=True)
        self.ds = xr.open_zarr(zarr_path)

        self.var_names = [
            'ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
            'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral'
        ]
        for i, v in enumerate(self.var_names):
            if v not in self.ds:
                short_v = v.replace('_icosahedral', '')
                if short_v in self.ds:
                    self.var_names[i] = short_v

        self.times = self.ds['time'].values
        self.num_samples = len(self.times)
        self.num_levels = self.ds.sizes.get('level', self.ds.sizes.get('height', 32))
        self.num_nodes = self.ds.sizes.get('node', 40962)
        self.num_vars = len(self.var_names)

        if 'latitude' in self.ds and 'longitude' in self.ds:
            self.latitudes = np.nan_to_num(self.ds['latitude'].values, nan=0.0)
            self.longitudes = np.nan_to_num(self.ds['longitude'].values, nan=0.0)
        else:
            self.latitudes = np.linspace(-90, 90, self.num_nodes)
            self.longitudes = np.linspace(-180, 180, self.num_nodes)

        if self.latitudes.ndim > 1:
            self.latitudes = self.latitudes[0]
        if self.longitudes.ndim > 1:
            self.longitudes = self.longitudes[0]
        self.longitudes = np.where(self.longitudes > 180.0, self.longitudes - 360.0, self.longitudes)

        # Load Static Surface Features
        self.h_terrain = self._extract_2d_surface_feature(['h_terrain_icosahedral', 'h_terrain', 'elevation'], default_val=0.0)
        self.land_sea_mask = self._extract_2d_surface_feature(['land_sea_mask'], default_val=0.0)

        # Compute Surface Roughness Length z0 (Ocean = 0.0002m, Land = 0.1m)
        z0_map = np.where(self.land_sea_mask > 0.5, 0.1, 0.0002).astype(np.float32)
        ln_z0_norm = (np.log(z0_map + 1e-5) / 5.0).astype(np.float32)

        # Build 3-channel base static topography [Elevation, LSM, z0]
        static_topo_raw = np.stack([
            self.h_terrain / 10000.0,
            self.land_sea_mask,
            ln_z0_norm
        ], axis=0)
        self.static_topo_base = np.nan_to_num(static_topo_raw, nan=0.0, posinf=1.0, neginf=-2.0).astype(np.float32)

    def _extract_2d_surface_feature(self, candidate_names: list, default_val: float = 0.0) -> np.ndarray:
        for name in candidate_names:
            if name in self.ds:
                arr = self.ds[name].values
                if arr.ndim > 1:
                    arr = arr[0]
                return np.nan_to_num(arr, nan=default_val, posinf=default_val, neginf=default_val).astype(np.float32)
        return np.full((self.num_nodes,), default_val, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def _load_observations_for_time(self, time_val, h_3d_profile: np.ndarray, p_3d_profile: np.ndarray):
        """Parses satellite and conventional observations for active loss hooks."""
        obs_dict = {
            'obs_amsua_tb': np.full((15, self.num_nodes), 240.0, dtype=np.float32),
            'obs_amsua_mask': np.zeros((15, self.num_nodes), dtype=np.float32),
            'obs_iasi_tb': np.full((30, self.num_nodes), 240.0, dtype=np.float32),
            'obs_iasi_mask': np.zeros((30, self.num_nodes), dtype=np.float32),
            'obs_hms_tb': np.full((12, self.num_nodes), 240.0, dtype=np.float32),
            'obs_hms_mask': np.zeros((12, self.num_nodes), dtype=np.float32),
            'obs_atms_tb': np.full((22, self.num_nodes), 240.0, dtype=np.float32),
            'obs_atms_mask': np.zeros((22, self.num_nodes), dtype=np.float32),
            'obs_cris_tb': np.full((30, self.num_nodes), 240.0, dtype=np.float32),
            'obs_cris_mask': np.zeros((30, self.num_nodes), dtype=np.float32),
            'obs_seviri_tb': np.full((8, self.num_nodes), 240.0, dtype=np.float32),
            'obs_seviri_mask': np.zeros((8, self.num_nodes), dtype=np.float32),
            'obs_gsrasr_tb': np.full((10, self.num_nodes), 240.0, dtype=np.float32),
            'obs_gsrasr_mask': np.zeros((10, self.num_nodes), dtype=np.float32),
            'obs_gsrcsr_tb': np.full((7, self.num_nodes), 240.0, dtype=np.float32),
            'obs_gsrcsr_mask': np.zeros((7, self.num_nodes), dtype=np.float32),
            'obs_ahicsr_tb': np.full((9, self.num_nodes), 240.0, dtype=np.float32),
            'obs_ahicsr_mask': np.zeros((9, self.num_nodes), dtype=np.float32),
            # Conventional Observation Containers [Vars=7, Levels=32, Nodes]
            'obs_conv_val': np.zeros((self.num_vars, self.num_levels, self.num_nodes), dtype=np.float32),
            'obs_conv_mask': np.zeros((self.num_vars, self.num_levels, self.num_nodes), dtype=np.float32),
        }

        if self.obs_dir and os.path.exists(self.obs_dir):
            dt_str = str(time_val)[:13].replace('-', '').replace('T', '.t') + 'z'

            # 1. Multi-Sensor Satellite Observations
            obs_file_pattern = os.path.join(self.obs_dir, f"obs_unified.*{dt_str}*.nc")
            matching_files = glob.glob(obs_file_pattern)

            if matching_files:
                try:
                    ds_obs = xr.open_dataset(matching_files[0])
                    vals = np.nan_to_num(ds_obs['observation_value'].values, nan=0.0)
                    sensors = ds_obs['sensor'].values
                    channels = ds_obs['channel'].values
                    lons = ds_obs['longitude'].values

                    node_idx = ((lons + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    node_idx = np.clip(node_idx, 0, self.num_nodes - 1)

                    sensor_specs = [
                        ('amsua', 15, 'obs_amsua_tb', 'obs_amsua_mask', 1),
                        ('iasi', 30, 'obs_iasi_tb', 'obs_iasi_mask', 1),
                        ('hms', 12, 'obs_hms_tb', 'obs_hms_mask', 1),
                        ('atms', 22, 'obs_atms_tb', 'obs_atms_mask', 1),
                        ('cris', 30, 'obs_cris_tb', 'obs_cris_mask', 1),
                        ('seviri', 8, 'obs_seviri_tb', 'obs_seviri_mask', 4),
                        ('gsrasr', 10, 'obs_gsrasr_tb', 'obs_gsrasr_mask', 7),
                        ('gsrcsr', 7, 'obs_gsrcsr_tb', 'obs_gsrcsr_mask', 8),
                        ('ahicsr', 9, 'obs_ahicsr_tb', 'obs_ahicsr_mask', 7),
                    ]

                    for sens_id, n_ch, tb_key, mask_key, ch_offset in sensor_specs:
                        mask = (sensors == sens_id) & (vals > 100.0) & (vals < 350.0)
                        if np.any(mask):
                            for c, v, n in zip(channels[mask], vals[mask], node_idx[mask]):
                                ch_idx = c - ch_offset
                                if 0 <= ch_idx < n_ch:
                                    obs_dict[tb_key][ch_idx, n] = v
                                    obs_dict[mask_key][ch_idx, n] = 1.0

                    ds_obs.close()
                except Exception:
                    pass

            # 2. Conventional Observations with Vertical p -> z Interpolation
            conv_pattern = os.path.join(self.obs_dir, f"obs_conv.*{dt_str}*.nc")
            conv_files = glob.glob(conv_pattern)

            if conv_files:
                try:
                    ds_conv = xr.open_dataset(conv_files[0])
                    c_lons = ds_conv['longitude'].values
                    c_pressures = ds_conv['pressure'].values
                    c_var_types = ds_conv['variable_type'].values
                    c_obs_vals = np.nan_to_num(ds_conv['observation_value'].values, nan=0.0)

                    c_node_idx = ((c_lons + 180.0) / 360.0 * (self.num_nodes - 1)).astype(int)
                    c_node_idx = np.clip(c_node_idx, 0, self.num_nodes - 1)

                    if np.nanmean(c_pressures) < 2000.0:
                        c_pressures = c_pressures * 100.0

                    for p_obs, v_type, val, n_idx in zip(c_pressures, c_var_types, c_obs_vals, c_node_idx):
                        if 0 <= v_type < self.num_vars and val != 0.0:
                            node_p_profile = p_3d_profile[:, n_idx]
                            if p_obs <= node_p_profile[0] and p_obs >= node_p_profile[-1]:
                                log_p_node = np.log(np.clip(node_p_profile, 1.0, None))
                                log_p_obs = np.log(np.clip(p_obs, 1.0, None))

                                k_idx = int(np.interp(-log_p_obs, -log_p_node, np.arange(self.num_levels)))
                                k_idx = np.clip(k_idx, 0, self.num_levels - 1)

                                obs_dict['obs_conv_val'][v_type, k_idx, n_idx] = val
                                obs_dict['obs_conv_mask'][v_type, k_idx, n_idx] = 1.0

                    ds_conv.close()
                except Exception:
                    pass

        clean_obs = {}
        for k, v in obs_dict.items():
            clean_v = np.nan_to_num(v, nan=0.0).astype(np.float32)
            clean_obs[k] = torch.from_numpy(clean_v)

        return clean_obs


class LogState4DForecastDataset(LogStateZarrDataset):
    """
    4D AI-Data Assimilation Dataset Loader.
    Provides Background B_0h, Observations O_0h + Masks M_0h, 3D terrain-following heights h_3d,
    and Target Ground Truth Analysis A_0h at time 0h.
    """
    def __init__(self, zarr_path: str, obs_dir: str = None):
        super().__init__(zarr_path=zarr_path, obs_dir=obs_dir)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        t_curr_idx = idx

        # Background state B_0h [7, 32, Nodes]
        b_zero = np.stack([self.ds[v].isel(time=t_curr_idx).values for v in self.var_names], axis=0)
        b_zero = np.nan_to_num(b_zero, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

        # Target Analysis A_0h [7, 32, Nodes]
        target_a0 = b_zero.copy()

        valid_mask_np = ~np.isnan(self.ds[self.var_names[0]].isel(time=t_curr_idx).values)
        valid_mask_np = np.nan_to_num(valid_mask_np, nan=False).astype(bool)

        # M6 Terrain-Following Geometric Heights Compute: h = Hmax - eta*(Hmax - Hterrain)
        if 'h_icosahedral' in self.ds:
            h_3d = self.ds['h_icosahedral'].isel(time=t_curr_idx).values
        elif 'eta' in self.ds:
            eta_vals = self.ds['eta'].values
            Hmax = 20000.0
            h_terrain_val = self.ds['h_terrain_icosahedral'].isel(time=t_curr_idx).values if 'h_terrain_icosahedral' in self.ds else self.h_terrain
            if h_terrain_val.ndim > 1:
                h_terrain_val = h_terrain_val[0]
            h_3d = Hmax - eta_vals[:, np.newaxis] * (Hmax - h_terrain_val[np.newaxis, :])
        else:
            baseline_h = np.linspace(2, 20000, self.num_levels, dtype=np.float32)
            h_3d = np.repeat(baseline_h[:, np.newaxis], self.num_nodes, axis=1)

        h_3d = np.nan_to_num(h_3d, nan=0.0, posinf=20000.0, neginf=0.0).astype(np.float32)

        # Extract 3D pressure profile p_3d (Pa)
        ln_p_3d = self.ds[self.var_names[6]].isel(time=t_curr_idx).values
        p_3d_pa = np.exp(np.nan_to_num(ln_p_3d, nan=10.0))

        # Dynamic Solar Zenith Angle Compute
        time_val = self.times[t_curr_idx]
        timestamp_unix = float(np.datetime64(time_val, 's').astype(int))
        cos_sza = compute_solar_zenith_angle(self.latitudes, self.longitudes, timestamp_unix)

        # Concatenate 3-channel base static topo + 1-channel cos(SZA) -> 4 channels
        static_topo = np.concatenate([self.static_topo_base, cos_sza[np.newaxis, :]], axis=0)

        item = {
            'background_0h': torch.from_numpy(b_zero),            # [7, 32, Nodes]
            'target_analysis_0h': torch.from_numpy(target_a0),     # [7, 32, Nodes]
            'valid_mask': torch.from_numpy(valid_mask_np),        # [32, Nodes]
            'h_3d': torch.from_numpy(h_3d),                       # [32, Nodes]
            'static_topo': torch.from_numpy(static_topo),         # [Static_Feats=4, Nodes]
        }

        # Ingest all satellite and conventional observations
        item.update(self._load_observations_for_time(self.times[t_curr_idx], h_3d_profile=h_3d, p_3d_profile=p_3d_pa))
        return item


class SyntheticAIDAStateDataset(Dataset):
    """Synthetic Dataset Generator for Dry Testing."""
    def __init__(self, num_samples: int = 100, num_nodes: int = 40962, num_levels: int = 32):
        super().__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_levels = num_levels

        self.data_x = np.random.randn(num_samples, 7, num_levels, num_nodes).astype(np.float32)
        self.data_y = self.data_x + 0.05 * np.random.randn(num_samples, 7, num_levels, num_nodes).astype(np.float32)

        baseline_h = np.linspace(2, 20000, num_levels, dtype=np.float32)
        self.h_3d = np.repeat(baseline_h[:, np.newaxis], num_nodes, axis=1)
        self.static_topo = np.random.randn(4, num_nodes).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'background_0h': torch.from_numpy(self.data_x[idx]),
            'target_analysis_0h': torch.from_numpy(self.data_y[idx]),
            'valid_mask': torch.ones((self.num_levels, self.num_nodes), dtype=torch.bool),
            'h_3d': torch.from_numpy(self.h_3d),
            'static_topo': torch.from_numpy(self.static_topo),
            'obs_conv_val': torch.randn(7, self.num_levels, self.num_nodes, dtype=torch.float32),
            'obs_conv_mask': torch.ones(7, self.num_levels, self.num_nodes, dtype=torch.float32),
            'obs_amsua_tb': torch.full((15, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_amsua_mask': torch.ones((15, self.num_nodes), dtype=torch.float32),
            'obs_iasi_tb': torch.full((30, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_iasi_mask': torch.ones((30, self.num_nodes), dtype=torch.float32),
            'obs_hms_tb': torch.full((12, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_hms_mask': torch.ones((12, self.num_nodes), dtype=torch.float32),
            'obs_atms_tb': torch.full((22, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_atms_mask': torch.ones((22, self.num_nodes), dtype=torch.float32),
            'obs_cris_tb': torch.full((30, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_cris_mask': torch.ones((30, self.num_nodes), dtype=torch.float32),
            'obs_seviri_tb': torch.full((8, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_seviri_mask': torch.ones((8, self.num_nodes), dtype=torch.float32),
            'obs_gsrasr_tb': torch.full((10, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_gsrasr_mask': torch.ones((10, self.num_nodes), dtype=torch.float32),
            'obs_gsrcsr_tb': torch.full((7, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_gsrcsr_mask': torch.ones((7, self.num_nodes), dtype=torch.float32),
            'obs_ahicsr_tb': torch.full((9, self.num_nodes), 240.0, dtype=torch.float32),
            'obs_ahicsr_mask': torch.ones((9, self.num_nodes), dtype=torch.float32),
        }
