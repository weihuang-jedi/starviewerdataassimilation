#!/usr/bin/env python3
"""
NMC B-Matrix Parameter Generator for GFS NetCDF Files
------------------------------------------------------
Calculates 3D background error statistics (variances and vertical correlations)
for temperature (t), winds (u, v, w), specific humidity (q), and pressure (p)
from 24h/48h forecast differences valid at matching verification times.
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist


def parse_args():
    """Parse command line arguments for date filtering, input files, and output destination."""
    parser = argparse.ArgumentParser(
        description="Compute NMC Background Error Covariances (B-Matrix) from NetCDF GFS forecasts."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/ncfiles",
        help="Directory containing the input GFS NetCDF files. Default: current directory",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="2021-01-01T00:00:00",
        help="Start time for verification window (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default="2025-12-31T18:00:00",
        help="End time for verification window (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bmatrix_nmc.nc",
        help="Path for output NetCDF B-matrix file. Default: bmatrix_nmc_gfs.nc",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="NMC tuning scale factor to adjust for forecast error growth. Default: 0.2",
    )
    return parser.parse_args()


# ==============================================================================
# MODULE 1: FILE PAIRING & INGESTION
# ==============================================================================
def find_nmc_forecast_pairs(data_dir: str, start_time: str, end_time: str):
    """
    Scans files in data_dir to find pairs valid at the same verification time
    with a 24-hour lead time difference (e.g., f048 and f024).
    """
    file_pattern = os.path.join(data_dir, "gfs.*.f*.nc")
    all_files = sorted(glob.glob(file_pattern))

    if not all_files:
        raise FileNotFoundError(f"No NetCDF files matching 'gfs.*.f*.nc' found in {data_dir}")

    print(f"Scanning {len(all_files)} files for valid forecast pairs...")
    
    file_metadata = []
    t_start = pd.Timestamp(start_time)
    t_end = pd.Timestamp(end_time)

    for fname in all_files:
        try:
            # Quick metadata read using xarray without loading full data
            with xr.open_dataset(fname, engine="netcdf4") as ds:
                init_time = pd.Timestamp(ds["time"].values)
                step_hrs = float(ds["step"].values) / 1e9 / 3600  # timedelta to hours
                valid_time = pd.Timestamp(ds["valid_time"].values)

                # Filter by verification window
                if t_start <= valid_time <= t_end:
                    file_metadata.append({
                        "filepath": fname,
                        "init_time": init_time,
                        "step": step_hrs,
                        "valid_time": valid_time
                    })
        except Exception as e:
            print(f"Warning: Could not read metadata from {fname}: {e}")

    df_meta = pd.DataFrame(file_metadata)
    if df_meta.empty:
        raise ValueError(f"No forecast files found within valid_time window [{t_start} to {t_end}].")

    # Group by valid_time and find pairs with 24h lead difference (e.g., step 48 vs 24)
    matched_pairs = []
    for vtime, group in df_meta.groupby("valid_time"):
        f24 = group[group["step"] == 24.0]
        f48 = group[group["step"] == 48.0]

        if not f24.empty and not f48.empty:
            matched_pairs.append({
                "valid_time": vtime,
                "file_f48": f48.iloc[0]["filepath"],
                "file_f24": f24.iloc[0]["filepath"],
            })

    print(f"Found {len(matched_pairs)} valid forecast pairs (f048 - f024) in the requested window.")
    return matched_pairs


# ==============================================================================
# MODULE 2: STATISTICAL COMPUTATION
# ==============================================================================
def compute_nmc_statistics(matched_pairs, alpha=0.2):
    """
    Computes 3D variance profiles and vertical correlation matrices 
    from forecast difference fields (f48 - f24).
    """
    variables = ["t", "u", "v", "w", "q", "p"]
    
    # Read grid dimensions from the first file
    sample_ds = xr.open_dataset(matched_pairs[0]["file_f48"])
    heights = sample_ds["height"].values
    lats = sample_ds["latitude"].values
    lons = sample_ds["longitude"].values
    sample_ds.close()

    n_lev, n_lat, n_lon = len(heights), len(lats), len(lons)
    n_pairs = len(matched_pairs)

    # Accumulation containers for mean differences and variance components
    diff_sum = {var: np.zeros((n_lev, n_lat, n_lon), dtype=np.float64) for var in variables}
    diff_sq_sum = {var: np.zeros((n_lev, n_lat, n_lon), dtype=np.float64) for var in variables}
    
    # Vertical cross-covariance accumulators: shape (n_lev, n_lev)
    vert_cov_sum = {var: np.zeros((n_lev, n_lev), dtype=np.float64) for var in variables}

    print("\nProcessing forecast difference pairs...")
    for idx, pair in enumerate(matched_pairs):
        print(f"[{idx+1}/{n_pairs}] Processing valid_time: {pair['valid_time']}")
        ds48 = xr.open_dataset(pair["file_f48"])
        ds24 = xr.open_dataset(pair["file_f24"])

        for var in variables:
            if var in ds48 and var in ds24:
                # Difference array: delta_x = x_T48 - x_T24
                d_x = ds48[var].values - ds24[var].values
                
                # Replace FillValues or NaNs with 0
                d_x = np.nan_to_num(d_x, nan=0.0)

                diff_sum[var] += d_x
                diff_sq_sum[var] += d_x**2

                # Compute horizontally-averaged vertical profile for covariance
                # Reshape (n_lev, n_lat * n_lon)
                d_x_flat = d_x.reshape(n_lev, -1)
                vert_cov_sum[var] += (d_x_flat @ d_x_flat.T) / (n_lat * n_lon)

        ds48.close()
        ds24.close()

    # Finalize variance and vertical correlations
    var_3d_dict = {}
    vert_corr_dict = {}

    for var in variables:
        # Sample Mean Difference (Model Drift)
        mean_diff = diff_sum[var] / n_pairs
        
        # Unbiased 3D Variance scaled by alpha
        var_3d = alpha * ((diff_sq_sum[var] / n_pairs) - (mean_diff**2))
        var_3d_dict[var] = np.maximum(var_3d, 1e-12)  # Avoid zero-variance

        # Vertical Correlation Matrix
        v_cov = vert_cov_sum[var] / n_pairs
        v_std = np.sqrt(np.diag(v_cov))
        v_std[v_std == 0] = 1.0
        v_corr = v_cov / np.outer(v_std, v_std)
        vert_corr_dict[var] = np.clip(v_corr, -1.0, 1.0)

    return heights, lats, lons, var_3d_dict, vert_corr_dict


# ==============================================================================
# MODULE 3: NETCDF EXPORTER
# ==============================================================================
def export_bmatrix_netcdf(output_path, heights, lats, lons, var_3d_dict, vert_corr_dict, n_pairs, alpha):
    """Saves the 3D variances and vertical correlation matrices into a structured NetCDF dataset."""
    
    data_vars = {}
    
    # 1. Store 3D Error Variances for each variable
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
                    "long_name": f"NMC Background Error Variance for {var.upper()}",
                    "units": units,
                },
            )

    # 2. Store 2D Vertical Correlation Matrices (height x height_ref)
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
            "title": "NMC Background Error Covariance (B-Matrix) Parameters",
            "institution": "NOAA / NCEP / Anemoi DA Environment",
            "method": "NMC Method (24h - 48h Forecast Differences at Matching Valid Times)",
            "num_forecast_pairs": str(n_pairs),
            "scaling_factor_alpha": str(alpha),
            "conventions": "CF-1.8",
        },
    )

    ds_b.to_netcdf(output_path, encoding={v: {"zlib": True, "complevel": 4} for v in ds_b.data_vars})
    print(f"\nSuccessfully generated B-Matrix NetCDF file: '{output_path}'")


# ==============================================================================
# MAIN PIPELINE ENTRYPOINT
# ==============================================================================
def main():
    args = parse_args()

    # Step 1: Discover paired forecast files in the window
    pairs = find_nmc_forecast_pairs(
        data_dir=args.data_dir,
        start_time=args.start_time,
        end_time=args.end_time
    )

    # Step 2: Compute NMC Variances and Correlations
    heights, lats, lons, var_3d, vert_corr = compute_nmc_statistics(
        matched_pairs=pairs,
        alpha=args.alpha
    )

    # Step 3: Export to CF-Compliant NetCDF
    export_bmatrix_netcdf(
        output_path=args.output,
        heights=heights,
        lats=lats,
        lons=lons,
        var_3d_dict=var_3d,
        vert_corr_dict=vert_corr,
        n_pairs=len(pairs),
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()
