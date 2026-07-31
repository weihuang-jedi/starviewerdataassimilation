#!/usr/bin/env python3
"""
3D-Var Multi-Variable Level Matrix Plotter
------------------------------------------
Generates a 5x7 panel grid:
  - Rows (5): Variables [t, p, u, v, q]
  - Columns (7): Levels [0, 5, 10, 15, 20, 25, 30]

Usage:
  python3 plot_3dvar_matrix.py gfs_5var_3dvar_analysis.nc
"""

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

VARS = ["t", "p", "u", "v", "q"]
LEVELS = [0, 5, 10, 15, 20, 25, 30]


def get_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(f"Could not find coordinates matching {candidates}")


def plot_increment_matrix(nc_file: str, output_png: str = "3dvar_increment_matrix.png"):
    print(f"Loading 3D-Var dataset: {nc_file}")
    ds = xr.open_dataset(nc_file)

    lon_name = get_coord_name(ds, ["longitude", "lon"])
    lat_name = get_coord_name(ds, ["latitude", "lat"])
    h_name = get_coord_name(ds, ["height", "level", "lev", "z"])

    lons = ds[lon_name].values
    lats = ds[lat_name].values
    heights = ds[h_name].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Compute increments for all variables beforehand
    inc_data = {}
    for var in VARS:
        ana_var = f"{var}_ana" if f"{var}_ana" in ds else f"{var}_analysis"
        bg_var = f"{var}_bg" if f"{var}_bg" in ds else f"{var}_background"

        if f"{var}_increment" in ds:
            inc = ds[f"{var}_increment"].astype(np.float32)
        elif ana_var in ds and bg_var in ds:
            inc = (ds[ana_var] - ds[bg_var]).astype(np.float32)
        else:
            print(f"Warning: Missing fields for variable '{var}'. Filling with zeros.")
            inc = xr.DataArray(np.zeros((len(heights), len(lats), len(lons))), dims=[h_name, lat_name, lon_name])

        inc_data[var] = inc

    # Setup 5 rows x 7 columns Grid
    n_rows = len(VARS)
    n_cols = len(LEVELS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(28, 15),
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.05, "hspace": 0.15},
    )

    print("Building 35-panel matrix figure...")

    for r, var in enumerate(VARS):
        inc = inc_data[var]
        
        # Calculate robust max bound across all 7 selected levels for uniform row scaling
        sub_levels = [l for l in LEVELS if l < len(heights)]
        max_val = np.nanmax(np.abs(inc.isel({h_name: sub_levels}).values))
        if max_val == 0 or np.isnan(max_val):
            max_val = 1e-4

        vmin, vmax = -max_val, max_val
        cmap = "RdBu_r" if var in ["t", "p", "q"] else "PuOr"

        for c, lvl in enumerate(LEVELS):
            ax = axes[r, c]

            if lvl >= len(heights):
                ax.text(0.5, 0.5, f"Level {lvl}\nOut of bounds", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue

            data_2d = inc.isel({h_name: lvl}).values

            # Map Features
            ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.5, alpha=0.7)
            ax.add_feature(cfeature.BORDERS.with_scale("110m"), linestyle=":", linewidth=0.3, alpha=0.5)

            # Filled Contours
            cf = ax.contourf(
                lon_grid,
                lat_grid,
                data_2d,
                levels=np.linspace(0.05*vmin, 0.05*vmax, 20),
               #levels=np.linspace(0.5*vmin, 0.5*vmax, 20),
               #levels=np.linspace(vmin, vmax, 15),
                cmap=cmap,
                extend="both",
                transform=ccrs.PlateCarree(),
            )

            # Titles & Axis Labels
            if r == 0:
                h_val = heights[lvl]
                ax.set_title(f"Level {lvl}\n({h_val:.0f})", fontsize=11, fontweight="bold")

            if c == 0:
                ax.text(
                    -0.15,
                    0.5,
                    f"Var: {var.upper()}",
                    va="center",
                    ha="right",
                    rotation="vertical",
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                )

        # Add single row colorbar on the far right for each variable
        cbar_ax = fig.add_axes([0.91, 0.74 - (r * 0.165), 0.012, 0.11])
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(f"Δ{var.upper()}", fontsize=10, fontweight="bold")

    fig.suptitle("3D-Var Analysis Increments Matrix Across Variables & Levels", fontsize=18, fontweight="bold", y=0.98)

    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Matrix plot created: '{output_png}'")
    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 5x7 3D-Var Increment Matrix")
    parser.add_argument("--input", type=str, default="cycling_output_20240106/aida_analysis_20240106_t00z.nc", help="Path to 3D-Var Analysis NetCDF file")
    parser.add_argument("--output", type=str, default="3dvar_increment_matrix.png", help="Output PNG path")

    args = parser.parse_args()
    plot_increment_matrix(args.input, args.output)
