#!/usr/bin/env python3
"""
utils/plot_aida_increments.py
-----------------------------
AIDA Diagnostic Plotter: 3x6 Multi-Variable Panel Plot for Native Icosahedral Grids.
Rows (Top-to-Bottom): Background (B0h), AIDA Analysis (A0h), Analysis Increment (A - B)
Columns (Left-to-Right): State Variables (t, p, u, v, w, q)
Uses tripcolor with masked anti-meridian crossing triangles to eliminate horizontal stripes across Asia/Pacific.
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AIDA 3x6 native icosahedral comparison panel.")
    parser.add_argument("-b", "--background", required=True, help="Path to Background Icosahedral NetCDF file")
    parser.add_argument("-a", "--analysis", required=True, help="Path to Analysis Icosahedral NetCDF file")
    parser.add_argument("-idx", "--height_idx", type=int, default=10, help="Vertical height level index (0 to 31, default: 10)")
    parser.add_argument("-o", "--output", default="output/plots/aida_panel_3x6_icosahedral.png", help="Path to save output figure")
    parser.add_argument("-s", "--show", action="store_true", help="Display plot interactively")
    return parser.parse_args()


def load_and_align_datasets(bg_path: str, an_path: str):
    """Loads background and analysis icosahedral NetCDF files."""
    if not os.path.exists(bg_path):
        raise FileNotFoundError(f"[ERROR] Background file not found: '{bg_path}'")
    if not os.path.exists(an_path):
        raise FileNotFoundError(f"[ERROR] Analysis file not found: '{an_path}'")

    ds_bg = xr.open_dataset(bg_path)
    ds_an = xr.open_dataset(an_path)
    return ds_bg, ds_an


def unnormalize_var_slice(var_name: str, raw_slice: np.ndarray) -> np.ndarray:
    """Converts log-state dynamic variables to true physical units."""
    mean_val = np.nanmean(raw_slice)
    
    if var_name == "t":
        if mean_val < 10.0:
            return np.exp(raw_slice)
        return raw_slice
    elif var_name == "p":
        if mean_val < 15.0:
            return np.exp(raw_slice) / 100.0
        elif mean_val > 10000.0:
            return raw_slice / 100.0
        return raw_slice
    elif var_name == "q":
        return np.maximum(0.0, raw_slice)
    return raw_slice


def main():
    args = parse_args()
    ds_bg, ds_an = load_and_align_datasets(args.background, args.analysis)

    # Extract unstructured node coordinates
    lons = ds_an['longitude'].values
    lats = ds_an['latitude'].values
    if lons.ndim > 1:
        lons = lons[0]
    if lats.ndim > 1:
        lats = lats[0]

    # Convert longitudes from [0, 360] to [-180, 180] for Cartopy PlateCarree projection
    lons_clean = np.where(lons > 180.0, lons - 360.0, lons)

    # Extract triangular face connectivity array (face_nodes)
    if 'face_nodes' in ds_an:
        triangles = ds_an['face_nodes'].values
    elif 'face_nodes' in ds_bg:
        triangles = ds_bg['face_nodes'].values
    else:
        raise KeyError("[ERROR] Required 'face_nodes' triangulation connectivity array missing from NetCDF files!")

    # -------------------------------------------------------------------------
    # CRITICAL FIX: MASK TRIANGLES CROSSING THE 180° ANTI-MERIDIAN (DATE LINE)
    # -------------------------------------------------------------------------
    tri_lons = lons_clean[triangles]  # [Num_Faces, 3]
    # If longitude span within a triangle exceeds 180 deg, it crosses the Date Line
    lon_diff_01 = np.abs(tri_lons[:, 0] - tri_lons[:, 1])
    lon_diff_12 = np.abs(tri_lons[:, 1] - tri_lons[:, 2])
    lon_diff_20 = np.abs(tri_lons[:, 2] - tri_lons[:, 0])

    crossing_mask = (lon_diff_01 > 180.0) | (lon_diff_12 > 180.0) | (lon_diff_20 > 180.0)

    # Build Matplotlib Triangulation and apply anti-meridian mask
    triangulation = mtri.Triangulation(lons_clean, lats, triangles=triangles)
    triangulation.set_mask(crossing_mask)

    # Determine vertical height level metadata
    if 'target_level' in ds_an:
        heights = ds_an['target_level'].values
        actual_height = float(heights[args.height_idx])
    elif 'h_icosahedral' in ds_an:
        heights = np.nanmean(ds_an['h_icosahedral'].values, axis=1)
        actual_height = float(heights[args.height_idx])
    else:
        actual_height = float(args.height_idx)

    # Variable mappings between shorthand names and NetCDF variable keys
    var_order = ["t", "p", "u", "v", "w", "q"]
    var_key_map = {
        "t": ["ln_t_icosahedral", "ln_t", "t"],
        "p": ["ln_p_icosahedral", "ln_p", "p"],
        "u": ["u_icosahedral", "u"],
        "v": ["v_icosahedral", "v"],
        "w": ["w_icosahedral", "w"],
        "q": ["q_icosahedral", "q"]
    }

    var_meta = {
        "t": {"name": "T", "long_name": "Temperature", "units": "K", "cmap": "turbo"},
        "p": {"name": "P", "long_name": "Pressure", "units": "hPa", "cmap": "viridis"},
        "u": {"name": "U", "long_name": "Zonal Wind", "units": "m s⁻¹", "cmap": "PuOr_r"},
        "v": {"name": "V", "long_name": "Meridional Wind", "units": "m s⁻¹", "cmap": "RdBu_r"},
        "w": {"name": "W", "long_name": "Vertical Velocity", "units": "Pa s⁻¹", "cmap": "seismic"},
        "q": {"name": "Q", "long_name": "Specific Humidity", "units": "kg kg⁻¹", "cmap": "YlGnBu"}
    }

    row_titles = [
        "Background (B0h)",
        "AIDA Analysis (A0h)",
        "Analysis Increment (A - B)"
    ]

    proj = ccrs.PlateCarree()

    # Create 3x6 subplot layout
    fig, axes = plt.subplots(
        3, 6, figsize=(24, 12),
        subplot_kw={'projection': proj},
        constrained_layout=True
    )

    print(f"[AIDA PLOT] Generating 3x6 Masked Native Icosahedral Panel (Height Index {args.height_idx} | Height: {actual_height:.1f} m)...", flush=True)

    for col_idx, var in enumerate(var_order):
        meta = var_meta[var]

        bg_key = next((k for k in var_key_map[var] if k in ds_bg), None)
        an_key = next((k for k in var_key_map[var] if k in ds_an), None)

        if not bg_key or not an_key:
            print(f"[WARNING] Variable '{var}' not found in NetCDF datasets. Skipping column {col_idx}...", flush=True)
            continue

        bg_raw = np.squeeze(ds_bg[bg_key].values[args.height_idx])
        an_raw = np.squeeze(ds_an[an_key].values[args.height_idx])

        bg_phys = unnormalize_var_slice(var, bg_raw)
        an_phys = unnormalize_var_slice(var, an_raw)
        inc_phys = an_phys - bg_phys

        vmin = min(np.nanmin(bg_phys), np.nanmin(an_phys))
        vmax = max(np.nanmax(bg_phys), np.nanmax(an_phys))

        max_abs_inc = np.nanmax(np.abs(inc_phys))
        if max_abs_inc == 0 or np.isnan(max_abs_inc):
            max_abs_inc = 0.1

        print(f" -> [{meta['name']}] B0h range: [{np.nanmin(bg_phys):.2f}, {np.nanmax(bg_phys):.2f}] | A0h range: [{np.nanmin(an_phys):.2f}, {np.nanmax(an_phys):.2f}] | Inc max: {max_abs_inc:.3e}", flush=True)

        # var_order = ["t", "p", "u", "v", "w", "q"]
        if var == 't':
            max_abs_inc = 5.0
        elif var == 'p':
            max_abs_inc = 10.0
        elif var == 'u':
            max_abs_inc = 10.0
        elif var == 'v':
            max_abs_inc = 10.0
        elif var == 'w':
            max_abs_inc = 1.0
            vmin = -2.0
            vmax =  2.0
        elif var == 'q':
            max_abs_inc = 0.005
            vmin = 0.0
            vmax = 0.02

        col_data = [
            {"data": bg_phys, "cmap": meta["cmap"], "vmin": vmin, "vmax": vmax},
            {"data": an_phys, "cmap": meta["cmap"], "vmin": vmin, "vmax": vmax},
            {"data": inc_phys, "cmap": "coolwarm", "vmin": -max_abs_inc, "vmax": max_abs_inc}
        ]

        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            p = col_data[row_idx]

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle=':', edgecolor='gray')

            # Render unstructured triangular mesh directly using tripcolor (Masked triangulation removes seam artifacts)
            tc = ax.tripcolor(
                triangulation, p["data"],
                cmap=p["cmap"], vmin=p["vmin"], vmax=p["vmax"],
                transform=proj, shading='flat'
            )

            if row_idx == 0:
                ax.set_title(f"{meta['long_name']} ({meta['name']})\n[{meta['units']}]", fontsize=13, fontweight='bold', pad=10)

            if col_idx == 0:
                ax.text(
                    -0.08, 0.5, row_titles[row_idx],
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    va='center', ha='right', rotation=90
                )

            if row_idx == 1:
                cbar = fig.colorbar(tc, ax=[axes[0, col_idx], axes[1, col_idx]], orientation='horizontal', pad=0.03, shrink=0.85)
                cbar.ax.tick_params(labelsize=8)
            elif row_idx == 2:
                cbar = fig.colorbar(tc, ax=ax, orientation='horizontal', pad=0.03, shrink=0.85)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label(f"Δ {meta['name']}", fontsize=9)

    fig.suptitle(
        f"AIDA AI-Data Assimilation Diagnostics — Native Icosahedral Mesh (Height Index: {args.height_idx} | Level: {actual_height:.0f} m)",
        fontsize=18, fontweight='bold', y=1.02
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

    ds_bg.close()
    ds_an.close()
    print(f"[AIDA SUCCESS] Saved Clean Native Icosahedral 3x6 Panel Plot to: '{args.output}'", flush=True)


if __name__ == "__main__":
    main()
