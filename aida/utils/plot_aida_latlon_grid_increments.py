#!/usr/bin/env python3
"""
AIDA Diagnostic Plotter: 3x6 Multi-Variable Panel Plot
Rows (Top-to-Bottom): Background, Analysis, Increment
Columns (Left-to-Right): State Variables (t, p, u, v, w, q)
Fixed: Longitude wrapping issue across Prime Meridian using add_cyclic_point.
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AIDA 3x6 transposed multi-variable comparison panel.")
    parser.add_argument("-b", "--background", required=True, help="Path to Background NetCDF file")
    parser.add_argument("-a", "--analysis", required=True, help="Path to Analysis NetCDF file")
    parser.add_argument("-idx", "--height_idx", type=int, default=10, help="Vertical height index (0 to 31, default: 10)")
    parser.add_argument("-o", "--output", default="output/plots/panel_3x6_transposed.png", help="Path to save output figure")
    return parser.parse_args()


def load_and_align_datasets(bg_path, an_path):
    """Loads background and analysis datasets, harmonizing coordinate names."""
    if not os.path.exists(bg_path):
        raise FileNotFoundError(f"Background file not found: {bg_path}")
    if not os.path.exists(an_path):
        raise FileNotFoundError(f"Analysis file not found: {an_path}")

    ds_bg = xr.open_dataset(bg_path)
    ds_an = xr.open_dataset(an_path)

    for ds in [ds_bg, ds_an]:
        if 'latitude' in ds.coords and 'lat' not in ds.coords:
            ds = ds.rename({'latitude': 'lat'})
        if 'longitude' in ds.coords and 'lon' not in ds.coords:
            ds = ds.rename({'longitude': 'lon'})

    return ds_bg, ds_an


def main():
    args = parse_args()
    ds_bg, ds_an = load_and_align_datasets(args.background, args.analysis)

    # Extract spatial coordinates
    lat_key = 'lat' if 'lat' in ds_an.coords else 'latitude'
    lon_key = 'lon' if 'lon' in ds_an.coords else 'longitude'
    lats = ds_an[lat_key].values
    lons = ds_an[lon_key].values

    # Determine height metadata
    heights = ds_an['height'].values if 'height' in ds_an else np.arange(32)
    max_idx = len(heights) - 1
    if args.height_idx < 0 or args.height_idx > max_idx:
        raise ValueError(f"Invalid height_idx {args.height_idx}. Must be between 0 and {max_idx}.")

    actual_height = float(heights[args.height_idx]) if 'height' in ds_an else args.height_idx

    # Variable plot metadata configurations
    var_order = ["t", "p", "u", "v", "w", "q"]
    var_meta = {
        "t": {"name": "T", "long_name": "Temperature", "units": "K", "cmap": "turbo"},
        "p": {"name": "P", "long_name": "Pressure", "units": "Pa", "cmap": "viridis"},
        "u": {"name": "U", "long_name": "Zonal Wind", "units": "m s⁻¹", "cmap": "PuOr_r"},
        "v": {"name": "V", "long_name": "Meridional Wind", "units": "m s⁻¹", "cmap": "RdBu_r"},
        "w": {"name": "W", "long_name": "Vertical Velocity", "units": "Pa s⁻¹", "cmap": "seismic"},
        "q": {"name": "Q", "long_name": "Specific Humidity", "units": "kg kg⁻¹", "cmap": "YlGnBu"}
    }

    row_titles = [
        "Background (t00z + 6h)",
        "AIDA Analysis (t06z)",
        "Analysis Increment (A - B)"
    ]

    proj = ccrs.PlateCarree()

    # Create 3x6 subplot layout (Wide aspect ratio for widescreen/paper layout)
    fig, axes = plt.subplots(
        3, 6, figsize=(28, 12),
        subplot_kw={'projection': proj},
        constrained_layout=True
    )

    print(f"[AIDA PLOT] Generating 3x6 Transposed Panel for Height Index {args.height_idx} ({actual_height:.1f} m)...")

    for col_idx, var in enumerate(var_order):
        meta = var_meta[var]

        # Extract 2D slices at specified height index
        bg_slice = np.squeeze(ds_bg[var].values[args.height_idx])
        an_slice = np.squeeze(ds_an[var].values[args.height_idx])
        inc_slice = an_slice - bg_slice

        # Fix Prime Meridian discontinuity: Add cyclic longitude wrapper
        bg_cyc, cyclic_lons = add_cyclic_point(bg_slice, coord=lons)
        an_cyc, _ = add_cyclic_point(an_slice, coord=lons)
        inc_cyc, _ = add_cyclic_point(inc_slice, coord=lons)

        lon_mesh, lat_mesh = np.meshgrid(cyclic_lons, lats)

        # Calculate dynamic color ranges per variable
        vmin = min(np.nanmin(bg_slice), np.nanmin(an_slice))
        vmax = max(np.nanmax(bg_slice), np.nanmax(an_slice))

        print(f"backgrnd: {var} min: {np.nanmin(bg_slice)}, max: {np.nanmax(bg_slice)}")
        print(f"analysis: {var} min: {np.nanmin(an_slice)}, max: {np.nanmax(an_slice)}")

        max_abs_inc = np.nanmax(np.abs(inc_slice))
        if max_abs_inc == 0 or np.isnan(max_abs_inc):
            max_abs_inc = 1.0

        col_data = [
            {"data": bg_cyc, "cmap": meta["cmap"], "vmin": vmin, "vmax": vmax},
            {"data": an_cyc, "cmap": meta["cmap"], "vmin": vmin, "vmax": vmax},
            {"data": inc_cyc, "cmap": "coolwarm", "vmin": -max_abs_inc, "vmax": max_abs_inc}
        ]

        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            p = col_data[row_idx]

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle=':', edgecolor='gray')

            # Render pcolormesh
            mesh = ax.pcolormesh(
                lon_mesh, lat_mesh, p["data"],
                cmap=p["cmap"], vmin=p["vmin"], vmax=p["vmax"],
                transform=proj, shading='auto'
            )

            # Column headers on top row (Variable name)
            if row_idx == 0:
                ax.set_title(f"{meta['long_name']} ({meta['name']})\n[{meta['units']}]", fontsize=13, fontweight='bold', pad=10)

            # Row headers on left-most column (Data State)
            if col_idx == 0:
                ax.text(
                    -0.08, 0.5, row_titles[row_idx],
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    va='center', ha='right', rotation=90
                )

            # Horizontal Colorbars per Subplot Panel
            if row_idx == 1:
                # Colorbar for State Fields (Background & Analysis) shared under Row 1
                cbar = fig.colorbar(mesh, ax=[axes[0, col_idx], axes[1, col_idx]], orientation='horizontal', pad=0.03, shrink=0.85)
                cbar.ax.tick_params(labelsize=8)
            elif row_idx == 2:
                # Colorbar for Increment Field under Row 2
                cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.03, shrink=0.85)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label(f"Δ {meta['name']}", fontsize=9)

    # Super Title
    fig.suptitle(
        f"AIDA Diagnostics — State & Increment Overview (Height Index: {args.height_idx} | Level: {actual_height:.0f} m)",
        fontsize=18, fontweight='bold', y=1.02
    )

    # Save output plot
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[AIDA PLOT] Saved Transposed 3x6 Panel Plot to: {args.output}")


if __name__ == "__main__":
    main()
