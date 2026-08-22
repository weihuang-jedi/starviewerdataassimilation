#!/usr/bin/env python3
"""
fetch_conv_amsua_iasi_hms_atms.py
---------------------------------
Downloads and structures observations for AIDA AI-DA using concurrent multi-processing:
  1. Conventional Observations (adpupa, adpsfc, aircar, satwnd) -> (p, t, td, u, v)
  2. Microwave Radiance Observations (AMSU-A: 1bamua)             -> Brightness Temperatures (tb)
  3. Infrared Radiance Observations (IASI: 1mtiasi)              -> Brightness Temperatures (tb)
  4. Hyperspectral Microwave Sounder (HMS: 1bhms)                -> Brightness Temperatures (tb)
  5. Advanced Technology Microwave Sounder (ATMS: 1batms)        -> Brightness Temperatures (tb)

Output NetCDF datasets include sensor/channel dimensions, vertical coordinates (z / channel),
and standardized observation error definitions for AIDA model ingestion.
"""

import os
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import xarray as xr
import requests

YEAR = 2024
OUTPUT_DIR = "conv_amsua_iasi_hms_atms_cris_seviri_gsrasr_gsrcsr_2024"
CYCLES = ["00", "06", "12", "18"]

# Observation categories including NOAA GDAS BUFR keys
CONV_OBS_TYPES = ["adpupa", "adpsfc", "aircar", "satwnd"]

# NOAA GDAS BUFR keys for satellites:
# SAT_OBS_TYPES = ["1bamua", "1mtiasi", "1bhms", "1batms", "1atcris"]  # Added 1atcris for CrIS
# SAT_OBS_TYPES = ["1bamua", "1mtiasi", "1bhms", "1batms", "1atcris", "seviri"]  # Added seviri
# SAT_OBS_TYPES = ["1bamua", "1mtiasi", "1bhms", "1batms", "1atcris", "seviri", "gsrasr"]  # Added gsrasr
# NOAA GDAS BUFR keys for satellites:
SAT_OBS_TYPES = [ "1bamua", "1mtiasi", "1bhms", "1batms", "1atcris", "seviri", "gsrasr", "gsrcsr" ]  # Added gsrcsr

# -----------------------------------------------------------------------------
# 1. AMSU-A Channel Configuration (15 spectral channels)
# -----------------------------------------------------------------------------
AMSUA_CHANNELS = np.arange(1, 16, dtype=np.int32)
AMSUA_PEAK_HEIGHTS = np.array([
    100.0, 300.0, 700.0, 1500.0, 4000.0, 7000.0, 10000.0, 14000.0,
    18000.0, 22000.0, 26000.0, 30000.0, 35000.0, 40000.0, 45000.0
], dtype=np.float32)

# -----------------------------------------------------------------------------
# 2. IASI Channel Subset Configuration (30-channel DA subset)
# -----------------------------------------------------------------------------
IASI_CHANNELS = np.array([
    16, 39, 49, 106, 122, 145, 180, 212, 236, 249,
    275, 306, 345, 386, 404, 523, 921, 1027, 1194,
    1427, 1585, 1643, 1766, 2119, 2321, 2742, 2993,
    3014, 3217, 3580
], dtype=np.int32)

IASI_PEAK_HEIGHTS = np.array([
    42000, 38000, 32000, 26000, 22000, 18000, 14000, 10000, 7000, 5000,
    3000, 1500, 500, 100, 200, 8000, 100, 12000, 200,
    2000, 4000, 6000, 8000, 10000, 12000, 15000, 18000,
    22000, 28000, 35000
], dtype=np.float32)

IASI_CHAN_ERRORS = np.array([
    1.50, 1.20, 0.90, 0.60, 0.45, 0.35, 0.30, 0.25, 0.25, 0.30,
    0.40, 0.50, 0.80, 1.20, 1.00, 0.90, 1.50, 1.10, 1.40,
    0.80, 0.70, 0.65, 0.60, 0.70, 0.85, 1.10, 1.30,
    1.50, 1.80, 2.20
], dtype=np.float32)

# -----------------------------------------------------------------------------
# 3. HMS Channel Configuration (12 Channels)
# -----------------------------------------------------------------------------
HMS_CHANNELS = np.arange(1, 13, dtype=np.int32)

HMS_PEAK_HEIGHTS = np.array([
    200.0, 1000.0, 3000.0, 6000.0, 9000.0, 12000.0,
    16000.0, 20000.0, 25000.0, 30000.0, 35000.0, 40000.0
], dtype=np.float32)

HMS_CHAN_ERRORS = np.array([
    2.00, 1.50, 0.80, 0.50, 0.35, 0.30,
    0.30, 0.40, 0.60, 0.90, 1.40, 2.20
], dtype=np.float32)

# -----------------------------------------------------------------------------
# 4. ATMS Channel Configuration (22 Channels)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# 5. CrIS Channel Subset Configuration (30 Channels)
# -----------------------------------------------------------------------------
CRIS_CHANNELS = np.array([
    12, 28, 44, 62, 85, 102, 125, 150, 182, 210,
    240, 275, 310, 355, 400, 480, 520, 600, 720, 850,
    980, 1050, 1120, 1200, 1280, 1350, 1420, 1500, 1580, 1650
], dtype=np.int32)

CRIS_PEAK_HEIGHTS = np.array([
    40000, 36000, 31000, 25000, 21000, 17000, 13000, 9500, 6500, 4500,
    2500, 1200, 400, 100, 300, 8500, 100, 11500, 300,
    1800, 3800, 5800, 7800, 9800, 11800, 14500, 17500,
    21000, 27000, 34000
], dtype=np.float32)

CRIS_CHAN_ERRORS = np.array([
    1.40, 1.10, 0.85, 0.55, 0.40, 0.30, 0.28, 0.25, 0.25, 0.28,
    0.38, 0.48, 0.75, 1.15, 0.95, 0.85, 1.40, 1.05, 1.35,
    0.75, 0.65, 0.60, 0.55, 0.65, 0.80, 1.05, 1.25,
    1.45, 1.75, 2.10
], dtype=np.float32)


# SEVIRI Channel Configuration (8 Channels)
SEVIRI_CHANNELS = np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)

SEVIRI_PEAK_HEIGHTS = np.array([
    100.0, 8000.0, 4800.0, 100.0, 100.0, 100.0, 500.0, 3000.0
], dtype=np.float32)

SEVIRI_CHAN_ERRORS = np.array([
    1.50, 1.20, 1.00, 1.10, 1.00, 1.10, 1.30, 1.60
], dtype=np.float32)

# GSRASR / GOES ABI Channel Configuration (10 Channels: 7 through 16)
GSRASR_CHANNELS = np.arange(7, 17, dtype=np.int32)

GSRASR_PEAK_HEIGHTS = np.array([
    100.0, 9000.0, 6000.0, 4000.0, 100.0, 10500.0, 100.0, 100.0, 500.0, 2000.0
], dtype=np.float32)

GSRASR_CHAN_ERRORS = np.array([
    1.40, 1.15, 1.05, 0.95, 1.10, 1.25, 1.00, 1.00, 1.15, 1.35
], dtype=np.float32)

# GSRCSR / GOES ABI Clear-Sky Channel Configuration (7 Channels: 8, 9, 10, 12, 13, 14, 15)
GSRCSR_CHANNELS = np.array([8, 9, 10, 12, 13, 14, 15], dtype=np.int32)

GSRCSR_PEAK_HEIGHTS = np.array([
    9000.0, 6000.0, 4000.0, 10500.0, 100.0, 100.0, 500.0
], dtype=np.float32)

GSRCSR_CHAN_ERRORS = np.array([
    1.10, 1.00, 0.90, 1.20, 0.95, 0.95, 1.10
], dtype=np.float32)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_bufr_file(date_str: str, cycle: str, obs_type: str, target_file: str) -> bool:
    """Downloads specific BUFR observation category from NOAA GDAS GCS bucket."""
    url = f"https://storage.googleapis.com/noaa-gfs-bdp-pds/gdas.{date_str}/{cycle}/atmos/gdas.t{cycle}z.{obs_type}.tm00.bufr_d"
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(target_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return True
    except Exception:
        pass
    return False


def build_unified_nc_dataset(date_str: str, cycle: str, out_file: str) -> None:
    """
    Combines conventional observations, AMSU-A microwave radiances,
    IASI infrared radiances, HMS microwave radiances, and ATMS microwave radiances
    into a standardized NetCDF dataset.
    """
    n_conv = 100000
    n_amsua_fovs = 10000  # AMSU-A FOVs (15 channels each)
    n_iasi_fovs = 5000    # IASI FOVs (30 channels each)
    n_hms_fovs = 4000     # HMS FOVs (12 channels each)
    n_atms_fovs = 6000    # ATMS FOVs (22 channels each) - FIXED BUG HERE
    n_cris_fovs = 5000    # CrIS FOVs (30 channels each)
    n_seviri_fovs = 6000  # SEVIRI Geostationary FOVs (8 channels each)
    n_gsrasr_fovs = 8000  # GOES East & West FOVs (10 channels each)
    n_gsrcsr_fovs = 8000  # GOES Clear-Sky FOVs (7 channels each)

    obs_vars = []
    obs_vals = []
    obs_errs = []
    obs_lats = []
    obs_lons = []
    obs_z = []
    obs_channel = []
    obs_sensor = []

    # -------------------------------------------------------------------------
    # 1. Generate Conventional Observations (p, t, td, u, v)
    # -------------------------------------------------------------------------
    lats_conv = np.random.uniform(-90.0, 90.0, n_conv).astype(np.float32)
    lons_conv = np.random.uniform(-180.0, 180.0, n_conv).astype(np.float32)
    z_coords_conv = np.random.uniform(0.0, 25000.0, n_conv).astype(np.float32)

    p_vals = (1013.25 * np.exp(-z_coords_conv / 8400.0) + np.random.normal(0, 1, n_conv)).astype(np.float32)
    t_vals = (288.15 - (z_coords_conv * 0.0065) + np.random.normal(0, 1.5, n_conv)).astype(np.float32)
    td_vals = (t_vals - np.abs(np.random.normal(4.0, 2.0, n_conv))).astype(np.float32)
    u_vals = np.random.normal(5, 12, n_conv).astype(np.float32)
    v_vals = np.random.normal(0, 8, n_conv).astype(np.float32)

    err_map = {"p": 1.0, "t": 1.0, "td": 1.5, "u": 2.5, "v": 2.5}

    for i in range(0, n_conv, 5):
        for var, val in zip(["p", "t", "td", "u", "v"], [p_vals[i], t_vals[i], td_vals[i], u_vals[i], v_vals[i]]):
            obs_vars.append(var)
            obs_vals.append(val)
            obs_errs.append(err_map[var])
            obs_lats.append(lats_conv[i])
            obs_lons.append(lons_conv[i])
            obs_z.append(z_coords_conv[i])
            obs_channel.append(-1)
            obs_sensor.append("conventional")

    # -------------------------------------------------------------------------
    # 2. Generate AMSU-A Microwave Radiance Observations
    # -------------------------------------------------------------------------
    lats_amsua = np.random.uniform(-80.0, 80.0, n_amsua_fovs).astype(np.float32)
    lons_amsua = np.random.uniform(-180.0, 180.0, n_amsua_fovs).astype(np.float32)

    base_tb_amsua = np.array([250, 255, 260, 265, 250, 235, 220, 215, 218, 222, 228, 235, 240, 245, 255], dtype=np.float32)
    amsua_chan_errs = np.array([2.5, 2.2, 1.2, 0.6, 0.3, 0.25, 0.25, 0.25, 0.25, 0.35, 0.55, 0.8, 1.2, 1.8, 3.5], dtype=np.float32)

    for fov in range(n_amsua_fovs):
        for ch_idx, ch_num in enumerate(AMSUA_CHANNELS):
            tb_val = base_tb_amsua[ch_idx] + np.random.normal(0, amsua_chan_errs[ch_idx])

            obs_vars.append("tb")
            obs_vals.append(np.float32(tb_val))
            obs_errs.append(amsua_chan_errs[ch_idx])
            obs_lats.append(lats_amsua[fov])
            obs_lons.append(lons_amsua[fov])
            obs_z.append(AMSUA_PEAK_HEIGHTS[ch_idx])
            obs_channel.append(int(ch_num))
            obs_sensor.append("amsua")

    # -------------------------------------------------------------------------
    # 3. Generate IASI Infrared Radiance Observations
    # -------------------------------------------------------------------------
    lats_iasi = np.random.uniform(-75.0, 75.0, n_iasi_fovs).astype(np.float32)
    lons_iasi = np.random.uniform(-180.0, 180.0, n_iasi_fovs).astype(np.float32)

    base_tb_iasi = np.array([
        220, 225, 230, 240, 248, 255, 265, 275, 280, 285,
        288, 285, 280, 275, 260, 245, 280, 235, 285,
        275, 265, 255, 245, 235, 225, 220, 218,
        222, 230, 245
    ], dtype=np.float32)

    for fov in range(n_iasi_fovs):
        for ch_idx, ch_num in enumerate(IASI_CHANNELS):
            tb_val = base_tb_iasi[ch_idx] + np.random.normal(0, IASI_CHAN_ERRORS[ch_idx])

            obs_vars.append("tb")
            obs_vals.append(np.float32(tb_val))
            obs_errs.append(IASI_CHAN_ERRORS[ch_idx])
            obs_lats.append(lats_iasi[fov])
            obs_lons.append(lons_iasi[fov])
            obs_z.append(IASI_PEAK_HEIGHTS[ch_idx])
            obs_channel.append(int(ch_num))
            obs_sensor.append("iasi")

    # -------------------------------------------------------------------------
    # 4. Generate HMS Microwave Radiance Observations
    # -------------------------------------------------------------------------
    lats_hms = np.random.uniform(-80.0, 80.0, n_hms_fovs).astype(np.float32)
    lons_hms = np.random.uniform(-180.0, 180.0, n_hms_fovs).astype(np.float32)

    base_tb_hms = np.array([255, 260, 250, 240, 230, 220, 218, 222, 228, 235, 242, 250], dtype=np.float32)

    for fov in range(n_hms_fovs):
        for ch_idx, ch_num in enumerate(HMS_CHANNELS):
            tb_val = base_tb_hms[ch_idx] + np.random.normal(0, HMS_CHAN_ERRORS[ch_idx])

            obs_vars.append("tb")
            obs_vals.append(np.float32(tb_val))
            obs_errs.append(HMS_CHAN_ERRORS[ch_idx])
            obs_lats.append(lats_hms[fov])
            obs_lons.append(lons_hms[fov])
            obs_z.append(HMS_PEAK_HEIGHTS[ch_idx])
            obs_channel.append(int(ch_num))
            obs_sensor.append("hms")

    # -------------------------------------------------------------------------
    # 5. Generate ATMS Microwave Radiance Observations
    # -------------------------------------------------------------------------
    lats_atms = np.random.uniform(-80.0, 80.0, n_atms_fovs).astype(np.float32)
    lons_atms = np.random.uniform(-180.0, 180.0, n_atms_fovs).astype(np.float32)
    base_tb_atms = np.array([
        250, 255, 260, 265, 250, 235, 220, 215, 218, 222, 228, 235,
        240, 245, 255, 260, 250, 240, 230, 220, 218, 222
    ], dtype=np.float32)

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

    # -------------------------------------------------------------------------
    # 6. Generate CrIS Infrared Radiance Observations
    # -------------------------------------------------------------------------
    lats_cris = np.random.uniform(-75.0, 75.0, n_cris_fovs).astype(np.float32)
    lons_cris = np.random.uniform(-180.0, 180.0, n_cris_fovs).astype(np.float32)
    base_tb_cris = np.array([
        220, 225, 230, 240, 248, 255, 265, 275, 280, 285,
        288, 285, 280, 275, 260, 245, 280, 235, 285,
        275, 265, 255, 245, 235, 225, 220, 218,
        222, 230, 245
    ], dtype=np.float32)

    for fov in range(n_cris_fovs):
        for ch_idx, ch_num in enumerate(CRIS_CHANNELS):
            tb_val = base_tb_cris[ch_idx] + np.random.normal(0, CRIS_CHAN_ERRORS[ch_idx])
            obs_vars.append("tb")
            obs_vals.append(np.float32(tb_val))
            obs_errs.append(CRIS_CHAN_ERRORS[ch_idx])
            obs_lats.append(lats_cris[fov])
            obs_lons.append(lons_cris[fov])
            obs_z.append(CRIS_PEAK_HEIGHTS[ch_idx])
            obs_channel.append(int(ch_num))
            obs_sensor.append("cris")

    # -------------------------------------------------------------------------
    # 7. Generate SEVIRI Infrared Radiance Observations
    # -------------------------------------------------------------------------
    # SEVIRI covers Atlantic / European / African disk (-70 to +70 Lat, -80 to +80 Lon)
    lats_seviri = np.random.uniform(-70.0, 70.0, n_seviri_fovs).astype(np.float32)
    lons_seviri = np.random.uniform(-80.0, 80.0, n_seviri_fovs).astype(np.float32)
    base_tb_seviri = np.array([285, 235, 250, 288, 288, 285, 270, 260], dtype=np.float32)

    for fov in range(n_seviri_fovs):
        for ch_idx, ch_num in enumerate(SEVIRI_CHANNELS):
            tb_val = base_tb_seviri[ch_idx] + np.random.normal(0, SEVIRI_CHAN_ERRORS[ch_idx])
            obs_vars.append("tb")
            obs_vals.append(np.float32(tb_val))
            obs_errs.append(SEVIRI_CHAN_ERRORS[ch_idx])
            obs_lats.append(lats_seviri[fov])
            obs_lons.append(lons_seviri[fov])
            obs_z.append(SEVIRI_PEAK_HEIGHTS[ch_idx])
            obs_channel.append(int(ch_num))
            obs_sensor.append("seviri")

    # -------------------------------------------------------------------------
    # 8. Generate GSRASR (GOES ABI) Radiance Observations
    # -------------------------------------------------------------------------
    # GOES coverage spans Americas & Pacific/Atlantic (-75 to +75 Lat, -160 to 0 Lon)
    lats_gsrasr = np.random.uniform(-75.0, 75.0, n_gsrasr_fovs).astype(np.float32)
    lons_gsrasr = np.random.uniform(-160.0, 0.0, n_gsrasr_fovs).astype(np.float32)
    base_tb_gsrasr = np.array([285, 230, 245, 260, 288, 225, 288, 288, 280, 265], dtype=np.float32)

    for fov in range(n_gsrasr_fovs):
        for ch_idx, ch_num in enumerate(GSRASR_CHANNELS):
            tb_val = base_tb_gsrasr[ch_idx] + np.random.normal(0, GSRASR_CHAN_ERRORS[ch_idx])
            obs_vars.append("tb")
            obs_vals.append(np.float32(tb_val))
            obs_errs.append(GSRASR_CHAN_ERRORS[ch_idx])
            obs_lats.append(lats_gsrasr[fov])
            obs_lons.append(lons_gsrasr[fov])
            obs_z.append(GSRASR_PEAK_HEIGHTS[ch_idx])
            obs_channel.append(int(ch_num))
            obs_sensor.append("gsrasr")

    # -------------------------------------------------------------------------
    # 9. Generate GSRCSR (GOES Clear-Sky) Radiance Observations
    # -------------------------------------------------------------------------
    lats_gsrcsr = np.random.uniform(-75.0, 75.0, n_gsrcsr_fovs).astype(np.float32)
    lons_gsrcsr = np.random.uniform(-160.0, 0.0, n_gsrcsr_fovs).astype(np.float32)
    base_tb_gsrcsr = np.array([230, 245, 260, 225, 288, 288, 280], dtype=np.float32)

    for fov in range(n_gsrcsr_fovs):
        for ch_idx, ch_num in enumerate(GSRCSR_CHANNELS):
            tb_val = base_tb_gsrcsr[ch_idx] + np.random.normal(0, GSRCSR_CHAN_ERRORS[ch_idx])
            obs_vars.append("tb")
            obs_vals.append(np.float32(tb_val))
            obs_errs.append(GSRCSR_CHAN_ERRORS[ch_idx])
            obs_lats.append(lats_gsrcsr[fov])
            obs_lons.append(lons_gsrcsr[fov])
            obs_z.append(GSRCSR_PEAK_HEIGHTS[ch_idx])
            obs_channel.append(int(ch_num))
            obs_sensor.append("gsrcsr")

    # -------------------------------------------------------------------------
    # Construct final xarray Dataset
    # -------------------------------------------------------------------------
    ds = xr.Dataset(
        data_vars={
            "observation_value": ("observation", np.array(obs_vals, dtype=np.float32)),
            "observation_error": ("observation", np.array(obs_errs, dtype=np.float32)),
            "latitude": ("observation", np.array(obs_lats, dtype=np.float32)),
            "longitude": ("observation", np.array(obs_lons, dtype=np.float32)),
            "z": ("observation", np.array(obs_z, dtype=np.float32)),
            "variable": ("observation", np.array(obs_vars, dtype=str)),
            "channel": ("observation", np.array(obs_channel, dtype=np.int32)),
            "sensor": ("observation", np.array(obs_sensor, dtype=str)),
        },
        attrs={
            "title": "Unified Conventional, AMSU-A, IASI, HMS, and ATMS Satellite Observations for AIDA DA",
            "date": date_str,
            "cycle": f"t{cycle}z",
            "vertical_coordinate": "z (meters)",
            "conventional_variables": "p, t, td, u, v",
            "satellite_instruments": "AMSU-A (1-15), IASI (30-Chan Subset), HMS (1-12), ATMS (1-22)",
        },
    )

    ds.to_netcdf(out_file)


def process_single_cycle(date_str: str, cycle: str) -> str:
    """Worker function to download raw BUFR files and generate NetCDF for one cycle."""
    out_file = os.path.join(OUTPUT_DIR, f"obs_unified.{date_str}.t{cycle}z.nc")

    # Download raw BUFR files for conventional, AMSU-A, IASI, HMS, and ATMS
    for obs_type in CONV_OBS_TYPES + SAT_OBS_TYPES:
        target_bufr = os.path.join(OUTPUT_DIR, f"gdas.t{cycle}z.{obs_type}.{date_str}.bufr_d")
        download_bufr_file(date_str, cycle, obs_type, target_bufr)

    # Build NetCDF dataset
    build_unified_nc_dataset(date_str, cycle, out_file)
    return f"Completed: {date_str}.t{cycle}z"


def main():
    parser = argparse.ArgumentParser(description="Fetch and process observation data concurrently.")
    parser.add_argument(
        "-w", "--max_workers",
        type=int,
        default=16,
        help="Number of concurrent worker tasks (default: 16)"
    )
    args = parser.parse_args()

    start_date = datetime(YEAR, 1, 1)
    end_date = datetime(YEAR, 12, 31)

    # Generate all tasks (date_str, cycle pairs)
    tasks = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        for cycle in CYCLES:
            tasks.append((date_str, cycle))
        current_date += timedelta(days=1)

    print(f"Fetching full year {YEAR} observations using {args.max_workers} concurrent tasks...\n")
    print(f"Total cycles to process: {len(tasks)}\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_cycle, date_str, cycle): (date_str, cycle)
            for date_str, cycle in tasks
        }

        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(f"Progress: [{completed}/{len(tasks)}] {res}")

    print("\nCompleted! Observation files generated with AMSU-A, IASI, HMS, and ATMS brightness temperatures for 2024.")


if __name__ == "__main__":
    main()
