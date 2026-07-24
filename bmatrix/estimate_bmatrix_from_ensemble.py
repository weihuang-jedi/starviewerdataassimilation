#!/usr/bin/env python3
"""
Estimate B-Matrix Statistics from GFS Ensemble Files
--------------------------------------------------
Infers ensemble background error statistics across matching GFS NetCDF files.

Usage:
  python3 estimate_bmatrix_from_ensemble.py "/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/ncfiles/gfs.2023100*.t*z.1p00.f000.nc"
"""

import glob
import sys
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

VAR_LIST = ["p", "t", "u", "v", "q"]


def Gaussian_corr(r, L_h):
    return np.exp(-0.5 * (r / L_h) ** 2)


def get_dim_name(ds, candidates):
    """Find matching dimension name from candidate list."""
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    raise KeyError(f"Could not find any of dimensions {candidates} in dataset.")


def estimate_bmatrix_from_files(file_pattern: str, output_b_path: str = "bmatrix_calibrated.nc"):
    # Expand glob pattern if passed as a string
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    print(f"Found {len(file_list)} matching files for ensemble statistics:")
    for f in file_list[:5]:
        print(f"  - {f}")
    if len(file_list) > 5:
        print(f"  ... and {len(file_list) - 5} more files.")

    print("\nLoading files into ensemble dataset...")
    # Open dataset using xarray's MFDataset to treat distinct date files as ensemble samples
    ds_ens = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim="member",
        parallel=False,
    )

    n_members = ds_ens.sizes["member"]
    print(f"Successfully stacked dataset with {n_members} ensemble samples.")

    # Determine vertical and spatial dimension names
    h_dim = get_dim_name(ds_ens, ["height", "lev", "level", "plev", "isobaricInhPa", "z"])
    lat_dim = get_dim_name(ds_ens, ["latitude", "lat"])
    lon_dim = get_dim_name(ds_ens, ["longitude", "lon"])

    n_heights = ds_ens.sizes[h_dim]

    b_ds_vars = {}

    for var in VAR_LIST:
        if var not in ds_ens:
            print(f"Warning: Variable '{var}' not found in dataset. Skipping...")
            continue

        print(f"\nProcessing variable: '{var}'...")
        da = ds_ens[var].astype(np.float64)

        # Drop single-time dimension if present
        if "time" in da.dims:
            da = da.squeeze("time", drop=True)

        # 1. Compute Ensemble Mean & Perturbations
        ens_mean = da.mean(dim="member")
        pert = da - ens_mean  # Shape: (member, height, lat, lon)

        # 2. Compute 3D Variance (sigma_b^2)
        var_3d = pert.var(dim="member", ddof=1).values  # Shape: (height, lat, lon)
        b_ds_vars[f"{var}_var"] = (
            [h_dim, lat_dim, lon_dim],
            var_3d.astype(np.float32),
            {"long_name": f"Background Error Variance for {var}"},
        )

        # 3. Area-Averaged Vertical Correlation Matrix (L_v)
        pert_flat = pert.values.reshape(n_members, n_heights, -1)
        pert_flat = np.transpose(pert_flat, (1, 0, 2)).reshape(n_heights, -1)

        cov_v = np.cov(pert_flat)
        std_v = np.sqrt(np.diag(cov_v))
        std_v[std_v == 0] = 1e-6

        corr_v = cov_v / np.outer(std_v, std_v)
        corr_v += np.eye(n_heights) * 1e-6  # Tikhonov regularization

        b_ds_vars[f"{var}_vert_corr"] = (
            [h_dim, f"{h_dim}_p"],
            corr_v.astype(np.float32),
            {"long_name": f"Vertical Error Correlation Matrix for {var}"},
        )

        # 4. Diagnose Horizontal Correlation Length Scale (L_h in grid units)
        mid_k = n_heights // 2
        slice_2d = pert[:, mid_k, :, :].values

        distances, correlations = [], []
        for lag in range(1, 10):
            c_x = np.corrcoef(slice_2d[:, :, :-lag].ravel(), slice_2d[:, :, lag:].ravel())[0, 1]
            c_y = np.corrcoef(slice_2d[:, :-lag, :].ravel(), slice_2d[:, lag:, :].ravel())[0, 1]
            distances.append(lag)
            correlations.append(0.5 * (c_x + c_y))

        try:
            popt, _ = curve_fit(Gaussian_corr, distances, correlations, p0=[3.0])
            l_h_fit = popt[0]
        except Exception:
            l_h_fit = 3.0  # Fallback

        print(f"  -> Diagnosed Horizontal Length Scale L_h: {l_h_fit:.2f} grid units")

    # Construct and save output dataset
    ds_b = xr.Dataset(
        data_vars=b_ds_vars,
        coords={
            h_dim: ds_ens[h_dim].values,
            f"{h_dim}_p": ds_ens[h_dim].values,
            lat_dim: ds_ens[lat_dim].values,
            lon_dim: ds_ens[lon_dim].values,
        },
        attrs={
            "title": "Calibrated B-Matrix Statistics from GFS Samples",
            "source_files": file_pattern,
            "num_samples": str(n_members),
        },
    )

    ds_b.to_netcdf(output_b_path)
    print(f"\nSuccessfully generated B-Matrix file: '{output_b_path}'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = "/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/ncfiles/gfs.2023100*.t*z.1p00.f000.nc"

    # out_file = sys.argv[2] if len(sys.argv) > 2 else "bmatrix_calibrated.nc"
    out_file = "bmatrix_calibrated.nc"
    estimate_bmatrix_from_files(pattern, out_file)
