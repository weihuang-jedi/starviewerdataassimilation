#!/usr/bin/env python3
"""
Analysis-Based Background Error Covariance (B-Matrix) Generator
--------------------------------------------------------------
Calculates 3D background error statistics (variances and vertical correlations)
for temperature (t), winds (u, v, w), specific humidity (q), and pressure (p)
from 6-hourly analysis files (e.g., gfs.*.f000.nc) using time-anomaly perturbations.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute B-Matrix parameters from a sequence of 6-hourly GFS Analysis NetCDF files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/ncfiles",
        help="Directory containing the GFS analysis files. Default: current directory",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="gfs.*.f000.nc",
        help="Glob pattern matching analysis NetCDF files. Default: 'gfs.*.f000.nc'",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="2021-01-01T00:00:00",
        help="Start time for analysis window (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default="2025-12-31T18:00:00",
        help="End time for analysis window (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bmatrix_from_gfs_analysis.nc",
        help="Output NetCDF file name. Default: 'bmatrix_analysis_gfs.nc'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.03,
        help="Scaling factor to convert synoptic variability into background error variance. Default: 0.03",
    )
    return parser.parse_args()


# ==============================================================================
# MODULE 1: FILE INGESTION & FILTERING
# ==============================================================================
def filter_analysis_files(data_dir: str, pattern: str, start_time: str, end_time: str):
    """Filters analysis NetCDF files within the specified date/time range."""
    search_path = os.path.join(data_dir, pattern)
    all_files = sorted(glob.glob(search_path))

    if not all_files:
        raise FileNotFoundError(f"No NetCDF files matching '{pattern}' found in {data_dir}")

    print(f"Scanning {len(all_files)} analysis files...")

    t_start = pd.Timestamp(start_time)
    t_end = pd.Timestamp(end_time)
    valid_files = []

    for fname in all_files:
        try:
            with xr.open_dataset(fname, engine="netcdf4") as ds:
                # Extract reference time or valid_time
                if "valid_time" in ds:
                    file_time = pd.Timestamp(ds["valid_time"].values)
                elif "time" in ds:
                    file_time = pd.Timestamp(ds["time"].values)
                else:
                    continue

                if t_start <= file_time <= t_end:
                    valid_files.append((file_time, fname))
        except Exception as e:
            print(f"Warning: Could not read {fname}: {e}")

    # Sort files chronologically
    valid_files.sort(key=lambda x: x[0])
    selected_filepaths = [f[1] for f in valid_files]

    if not selected_filepaths:
        raise ValueError(f"No analysis files found within the window [{t_start} to {t_end}].")

    print(f"Selected {len(selected_filepaths)} analysis files for processing.")
    return selected_filepaths


# ==============================================================================
# MODULE 2: STATISTICAL COMPUTATION (ANOMALY METHOD)
# ==============================================================================
def compute_analysis_statistics(filepaths, alpha=0.03):
    """
    Calculates mean state, time-anomalies, 3D variances, and vertical correlation
    matrices across height levels for each 3D variable.
    """
    variables = ["t", "u", "v", "w", "q", "p"]
    
    # Read grid dimensions from the first file
    with xr.open_dataset(filepaths[0]) as sample_ds:
        heights = sample_ds["height"].values
        lats = sample_ds["latitude"].values
        lons = sample_ds["longitude"].values

    n_lev, n_lat, n_lon = len(heights), len(lats), len(lons)
    n_files = len(filepaths)

    # Accumulators for Mean calculation
    sum_dict = {var: np.zeros((n_lev, n_lat, n_lon), dtype=np.float64) for var in variables}

    print("\n[Pass 1/2] Computing temporal mean state across analyses...")
    for idx, fname in enumerate(filepaths):
        with xr.open_dataset(fname) as ds:
            for var in variables:
                if var in ds:
                    data = np.nan_to_num(ds[var].values, nan=0.0)
                    sum_dict[var] += data

    # Compute Mean fields
    mean_dict = {var: sum_dict[var] / n_files for var in variables}

    # Accumulators for Variance and Vertical Cross-Covariance
    sq_diff_sum = {var: np.zeros((n_lev, n_lat, n_lon), dtype=np.float64) for var in variables}
    vert_cov_sum = {var: np.zeros((n_lev, n_lev), dtype=np.float64) for var in variables}

    print("[Pass 2/2] Computing anomalies, 3D variances, and vertical correlation matrices...")
    for idx, fname in enumerate(filepaths):
        with xr.open_dataset(fname) as ds:
            for var in variables:
                if var in ds:
                    data = np.nan_to_num(ds[var].values, nan=0.0)
                    # Compute anomaly: delta_x = x_a(t) - mean_x
                    anomaly = data - mean_dict[var]

                    # 3D squared diff accumulator
                    sq_diff_sum[var] += anomaly**2

                    # Vertical correlation accumulator (averaged over horizontal domain)
                    anom_flat = anomaly.reshape(n_lev, -1)  # (n_lev, n_lat * n_lon)
                    vert_cov_sum[var] += (anom_flat @ anom_flat.T) / (n_lat * n_lon)

    # Post-process into scaled variances and correlation matrices
    var_3d_dict = {}
    vert_corr_dict = {}

    for var in variables:
        # Unbiased 3D variance scaled by alpha
        raw_var = sq_diff_sum[var] / max(n_files - 1, 1)
        var_3d = alpha * raw_var
        var_3d_dict[var] = np.maximum(var_3d, 1e-12)  # Prevent non-zero boundary issues

        # Vertical Correlation (-1 to +1)
        v_cov = vert_cov_sum[var] / n_files
        v_std = np.sqrt(np.diag(v_cov))
        v_std[v_std == 0] = 1.0  # Avoid division by zero
        v_corr = v_cov / np.outer(v_std, v_std)
        vert_corr_dict[var] = np.clip(v_corr, -1.0, 1.0)

    return heights, lats, lons, var_3d_dict, vert_corr_dict


# ==============================================================================
# MODULE 3: NETCDF EXPORTER
# ==============================================================================
def export_bmatrix_netcdf(output_path, heights, lats, lons, var_3d_dict, vert_corr_dict, n_files, alpha):
    """Saves the calculated 3D variances and 2D vertical correlations into a NetCDF file."""
    
    data_vars = {}
    
    # Metadata map
    var_metadata = {
        "t": ("temperature_error_variance", "K^2"),
        "u": ("u_wind_error_variance", "m2 s-2"),
        "v": ("v_wind_error_variance", "m2 s-2"),
        "w": ("vertical_velocity_error_variance", "Pa2 s-2"),
        "q": ("specific_humidity_error_variance", "kg2 kg-2"),
        "p": ("pressure_error_variance", "hPa2"),
    }

    for var, (std_name, units) in var_metadata.items():
        if var in var_3d_dict:
            data_vars[f"{var}_var"] = (
                ["height", "latitude", "longitude"],
                var_3d_dict[var].astype(np.float32),
                {
                    "standard_name": std_name,
                    "long_name": f"Analysis-Based Background Error Variance for {var.upper()}",
                    "units": units,
                },
            )

    # Add Vertical Correlation Matrices
    for var in var_3d_dict.keys():
        if var in vert_corr_dict:
            data_vars[f"{var}_vert_corr"] = (
                ["height", "height_ref"],
                vert_corr_dict[var].astype(np.float32),
                {
                    "long_name": f"Vertical Background Error Correlation Matrix for {var.upper()}",
                    "units": "1",
                },
            )

    ds_b = xr.Dataset(
        data_vars=data_vars,
        coords={
            "height": ("height", heights, {"units": "meters", "long_name": "Geometric Height Above Sea Level"}),
            "height_ref": ("height_ref", heights, {"units": "meters", "long_name": "Reference Geometric Height"}),
            "latitude": ("latitude", lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": ("longitude", lons, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "title": "Analysis-Based Background Error Covariance (B-Matrix) Parameters",
            "institution": "NOAA / NCEP / Anemoi DA Environment",
            "method": "Temporal Anomaly Perturbations from 6-Hourly Analysis Sequence",
            "num_analysis_samples": str(n_files),
            "scaling_factor_alpha": str(alpha),
            "conventions": "CF-1.8",
        },
    )

    # Write out NetCDF file with compression
    ds_b.to_netcdf(output_path, encoding={v: {"zlib": True, "complevel": 4} for v in ds_b.data_vars})
    print(f"\nSuccessfully generated B-Matrix NetCDF file: '{output_path}'")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    args = parse_args()

    # Step 1: Filter analysis files by time range
    selected_files = filter_analysis_files(
        data_dir=args.data_dir,
        pattern=args.file_pattern,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    # Step 2: Compute statistics from analysis sequence
    heights, lats, lons, var_3d, vert_corr = compute_analysis_statistics(
        filepaths=selected_files,
        alpha=args.alpha,
    )

    # Step 3: Export output NetCDF file
    export_bmatrix_netcdf(
        output_path=args.output,
        heights=heights,
        lats=lats,
        lons=lons,
        var_3d_dict=var_3d,
        vert_corr_dict=vert_corr,
        n_files=len(selected_files),
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
