#!/usr/bin/env python3
"""
3D-Var Matrix Plotter for Unstructured Icosahedral NetCDF Files.
Safely handles non-finite values (NaN/Inf), constant field slices, and unstructured triangulation.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def inspect_netcdf_sanitizations(ds: xr.Dataset):
    """
    Prints diagnostic summary of missing/non-finite values across target variables.
    """
    print("\n--- [AIDA DIAGNOSTICS] NetCDF Field Integrity Check ---")
    target_vars = [v for v in ds.data_vars if "icosahedral" in v or "ln_" in v]
    if not target_vars:
        target_vars = list(ds.data_vars.keys())

    for v in target_vars:
        arr = ds[v].values
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        total = arr.size
        min_val = np.nanmin(arr) if total > n_nan else np.nan
        max_val = np.nanmax(arr) if total > n_nan else np.nan
        print(
            f"Variable: {v:25s} | Shape: {str(arr.shape):15s} | "
            f"NaNs: {n_nan:6d} ({(n_nan/total)*100:.1f}%) | "
            f"Infs: {n_inf:6d} | Min: {min_val:.4f} | Max: {max_val:.4f}"
        )
    print("-------------------------------------------------------\n")


def get_mesh_coordinates(ds: xr.Dataset):
    """
    Extracts longitude and latitude arrays, handling 1D unstructured node arrays.
    Converts longitudes to standard [-180, 180] range if required.
    """
    lon_keys = ["lon", "longitude", "grid_lon", "lons"]
    lat_keys = ["lat", "latitude", "grid_lat", "lats"]

    lon, lat = None, None

    for k in lon_keys:
        if k in ds.coords or k in ds.data_vars:
            lon = ds[k].values
            break

    for k in lat_keys:
        if k in ds.coords or k in ds.data_vars:
            lat = ds[k].values
            break

    if lon is None or lat is None:
        raise KeyError(f"Could not locate 1D coordinate variables for mesh nodes in: {list(ds.keys())}")

    lon = lon.flatten()
    lat = lat.flatten()

    # Convert [0, 360] to [-180, 180] for standard map projections
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    return lon, lat


def safe_tricontourf(ax, lon, lat, data_sub, levels=15, cmap="viridis", title_str=""):
    """
    Renders tricontourf with explicit NaN/Inf filtering and constant field padding.
    """
    lon_flat = np.asarray(lon).flatten()
    lat_flat = np.asarray(lat).flatten()
    z_flat = np.asarray(data_sub).flatten()

    # 1. Filter out non-finite points
    valid_mask = np.isfinite(z_flat) & np.isfinite(lon_flat) & np.isfinite(lat_flat)

    if not np.any(valid_mask):
        ax.text(0.5, 0.5, "ALL-NAN SLICE", ha="center", va="center", transform=ax.transAxes, color="red", fontsize=9, weight="bold")
        ax.set_title(title_str, fontsize=8)
        return None

    lon_clean = lon_flat[valid_mask]
    lat_clean = lat_flat[valid_mask]
    z_clean = z_flat[valid_mask]

    # Need at least 3 distinct non-collinear points for Delaunay triangulation
    if len(z_clean) < 3:
        ax.text(0.5, 0.5, "< 3 VALID POINTS", ha="center", va="center", transform=ax.transAxes, color="orange", fontsize=8)
        ax.set_title(title_str, fontsize=8)
        return None

    # 2. Check value range to prevent zero-range division error in tricontourf
    vmin, vmax = np.min(z_clean), np.max(z_clean)
    if np.isclose(vmin, vmax):
        vmin -= 1e-4
        vmax += 1e-4

    lev_bounds = np.linspace(vmin, vmax, levels)

    # 3. Create triangulation and plot
    try:
        triang = tri.Triangulation(lon_clean, lat_clean)
        cf = ax.tricontourf(triang, z_clean, levels=lev_bounds, cmap=cmap, extend="both")
        ax.set_title(title_str, fontsize=8)
        return cf
    except Exception as err:
        ax.text(0.5, 0.5, f"Triangulation Error:\n{type(err).__name__}", ha="center", va="center", transform=ax.transAxes, color="red", fontsize=7)
        ax.set_title(title_str, fontsize=8)
        return None


def plot_analysis_matrix(nc_path: str, output_path: str):
    """
    Generates multi-level diagnostic analysis matrix plot.
    """
    print(f"[AIDA PLOT] Opening NetCDF analysis file: {nc_path}")
    ds = xr.open_dataset(nc_path)

    # Print data health diagnostic report
    inspect_netcdf_sanitizations(ds)

    # Extract mesh topology coordinates (2562 nodes)
    lon, lat = get_mesh_coordinates(ds)
    num_nodes = len(lon)

    # Identify target variables to display
    plot_vars = [v for v in ["ln_t_icosahedral", "ln_p_icosahedral", "ln_rho_icosahedral"] if v in ds.data_vars]
    if not plot_vars:
        plot_vars = [v for v in ds.data_vars if ds[v].ndim >= 2][:3]

    # Select level slices dynamically matching grid nodes dimension
    sample_var = ds[plot_vars[0]].values
    if sample_var.ndim >= 2:
        node_axis = [i for i, dim in enumerate(sample_var.shape) if dim == num_nodes]
        if node_axis:
            level_axis = 1 if node_axis[0] == 0 else 0
            n_levels = sample_var.shape[level_axis]
            level_indices = np.linspace(0, n_levels - 1, 5, dtype=int)
        else:
            level_indices = [0]
    else:
        level_indices = [0]

    num_rows = len(level_indices)
    num_cols = len(plot_vars)

    print(f"[AIDA PLOT] Generating {num_rows}x{num_cols} matrix visualization...")
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(4 * num_cols, 2.5 * num_rows), sharex=True, sharey=True, squeeze=False
    )

    for r, l_idx in enumerate(level_indices):
        for c, var_name in enumerate(plot_vars):
            ax = axes[r, c]
            field_data = ds[var_name].values

            # Robust slice selection to guarantee output shape of (num_nodes,) -> (2562,)
            if field_data.ndim == 2:
                if field_data.shape[1] == num_nodes:
                    slice_data = field_data[l_idx, :]
                else:
                    slice_data = field_data[:, l_idx]
            elif field_data.ndim == 3:
                if field_data.shape[2] == num_nodes:
                    slice_data = field_data[0, l_idx, :]
                else:
                    slice_data = field_data[0, :, l_idx]
            else:
                slice_data = field_data.flatten()

            title = f"{var_name} (L={l_idx})"
            cf = safe_tricontourf(ax, lon, lat, slice_data, levels=15, cmap="viridis", title_str=title)

            if cf is not None:
                plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, aspect=12)

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)

            if r == num_rows - 1:
                ax.set_xlabel("Longitude")
            if c == 0:
                ax.set_ylabel(f"L{l_idx}\nLatitude")

    plt.suptitle(f"AIDA GNN Cycling State Field Matrix\nFile: {Path(nc_path).name}", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[AIDA SUCCESS] Matrix plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AIDA 3D-Var Matrix Triangulation Plotter")
    parser.add_argument("--input", type=str, default="output/global_icosahedral_m4.20250106.t06z.1p00.anal.nc", help="Input analysis NetCDF path")
    parser.add_argument("--output", type=str, default="output/aida_3dvar_matrix.png", help="Output PNG file path")
    args = parser.parse_args()

    plot_analysis_matrix(args.input, args.output)


if __name__ == "__main__":
    main()
