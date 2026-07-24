#!/usr/bin/env python3
"""
3-Panel Diagnostic Plotter for NWP 3D-Var Data Assimilation
------------------------------------------------------------
Visualizes:
  1. Background Forecast State (x_b)
  2. 3D-Var Analysis State (x_a)
  3. Analysis Increment (delta_x = x_a - x_b)

Usage:
  python plot_da_analysis.py -i gfs.20230701.t12z.1p00.anal.nc -o analysis_plot.png --level_idx 0
"""

import argparse
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# ==============================================================================
# 1. ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 3D-Var Background, Analysis, and Increments in a 3-panel figure."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="gfs.20230701.t12z.1p00.anal.nc",
        help="Path to input NetCDF analysis file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="da_3panel_plot.png",
        help="Path for output PNG/PDF image file. Default: 'da_3panel_plot.png'",
    )
    parser.add_argument(
        "--var_prefix",
        type=str,
        default="t",
        help="Variable prefix in NetCDF (e.g., 't', 'u', 'v'). Default: 't'",
    )
    parser.add_argument(
        "--level_idx",
        type=int,
        default=10,
        help="Vertical height level index to plot (0 to N_heights-1). Default: 0",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for output figure. Default: 300",
    )
    return parser.parse_args()


# ==============================================================================
# 2. DATA LOADING MODULE
# ==============================================================================
def load_analysis_fields(file_path: str, var_prefix: str, level_idx: int):
    """Loads background, analysis, and increment arrays for a specific vertical level."""
    print(f"[1/2] Reading dataset: {file_path}")
    ds = xr.open_dataset(file_path)

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    heights = ds["height"].values

    if level_idx < 0 or level_idx >= len(heights):
        raise ValueError(
            f"Requested level_idx {level_idx} is out of bounds (0 to {len(heights)-1})."
        )

    height_val = heights[level_idx]

    # Extract slice at target height level
    x_b = ds[f"{var_prefix}_background"].isel(height=level_idx).values
    x_a = ds[f"{var_prefix}_analysis"].isel(height=level_idx).values
    dx = ds[f"{var_prefix}_increment"].isel(height=level_idx).values

    units = ds[f"{var_prefix}_background"].attrs.get("units", "K")

    return lats, lons, height_val, x_b, x_a, dx, units


# ==============================================================================
# 3. MODULAR PLOTTING ENGINE
# ==============================================================================
def plot_3panel_diagnostic(
    lats: np.ndarray,
    lons: np.ndarray,
    height_val: float,
    x_b: np.ndarray,
    x_a: np.ndarray,
    dx: np.ndarray,
    units: str,
    var_prefix: str,
    output_path: str,
    dpi: int = 300,
):
    """Generates a 3-row panel plot on global PlateCarree map projection."""
    print(f"[2/2] Generating 3-panel plot for Level Index height = {height_val:.1f}...")

    # Grid setup for cartopy
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 14),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    # --------------------------------------------------------------------------
    # Subplot 1: Background State (x_b)
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax1.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

    v_min = min(np.nanmin(x_b), np.nanmin(x_a))
    v_max = max(np.nanmax(x_b), np.nanmax(x_a))

    mesh1 = ax1.pcolormesh(
        lon_grid, lat_grid, x_b, cmap="coolwarm", vmin=v_min, vmax=v_max, transform=proj
    )
    cbar1 = fig.colorbar(mesh1, ax=ax1, orientation="vertical", shrink=0.8, pad=0.02)
    cbar1.set_label(f"[{units}]")
    ax1.set_title(
        f"(a) Background State ($x_b$) — {var_prefix.upper()} | Level: {height_val:.1f}\n"
        f"Min: {np.nanmin(x_b):.2f} | Mean: {np.nanmean(x_b):.2f} | Max: {np.nanmax(x_b):.2f} {units}",
        fontsize=11,
        loc="left",
    )

    # --------------------------------------------------------------------------
    # Subplot 2: Analysis State (x_a)
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax2.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

    mesh2 = ax2.pcolormesh(
        lon_grid, lat_grid, x_a, cmap="coolwarm", vmin=v_min, vmax=v_max, transform=proj
    )
    cbar2 = fig.colorbar(mesh2, ax=ax2, orientation="vertical", shrink=0.8, pad=0.02)
    cbar2.set_label(f"[{units}]")
    ax2.set_title(
        f"(b) Analysis State ($x_a$) — {var_prefix.upper()} | Level: {height_val:.1f}\n"
        f"Min: {np.nanmin(x_a):.2f} | Mean: {np.nanmean(x_a):.2f} | Max: {np.nanmax(x_a):.2f} {units}",
        fontsize=11,
        loc="left",
    )

    # --------------------------------------------------------------------------
    # Subplot 3: Analysis Increment (delta_x = x_a - x_b)
    # --------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax3.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

    # Symmetric color range centered at zero for increments
    max_inc = max(abs(np.nanmin(dx)), abs(np.nanmax(dx)))
    if max_inc == 0:
        max_inc = 1.0  # Prevent zero-division edge case

    mesh3 = ax3.pcolormesh(
        lon_grid,
        lat_grid,
        dx,
        cmap="RdBu_r",
        vmin=-max_inc,
        vmax=max_inc,
        transform=proj,
    )
    cbar3 = fig.colorbar(mesh3, ax=ax3, orientation="vertical", shrink=0.8, pad=0.02)
    cbar3.set_label(f"Increment [{units}]")
    ax3.set_title(
        f"(c) Analysis Increment ($\delta x = x_a - x_b$)\n"
        f"Min: {np.nanmin(dx):.4f} | Mean: {np.nanmean(dx):.4f} | Max: {np.nanmax(dx):.4f} {units}",
        fontsize=11,
        loc="left",
        weight="bold",
    )

    # Save output image
    fig.suptitle("GFS 3D-Var Data Assimilation Diagnostics", fontsize=14, weight="bold", y=0.995)
    plt.show()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[✓] Figure saved successfully: '{output_path}'")


# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def main():
    args = parse_args()

    # 1. Ingest Data
    lats, lons, height_val, x_b, x_a, dx, units = load_analysis_fields(
        file_path=args.input,
        var_prefix=args.var_prefix,
        level_idx=args.level_idx,
    )

    # 2. Render Figure
    plot_3panel_diagnostic(
        lats=lats,
        lons=lons,
        height_val=height_val,
        x_b=x_b,
        x_a=x_a,
        dx=dx,
        units=units,
        var_prefix=args.var_prefix,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
