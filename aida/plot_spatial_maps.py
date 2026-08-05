#!/usr/bin/env python3
"""
plot_spatial_maps.py
--------------------
Generates 2D global spatial maps for analysis increments and error distributions
using Cartopy and Matplotlib.
"""

import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def create_spatial_plot(lon, lat, field, title, colorbar_label, output_path, cmap='RdBu_r', vmin=None, vmax=None, symmetric=False):
    """
    Renders a global 2D latitude-longitude field on a PlateCarree projection.
    """
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add geographic context features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='gray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    # Symmetric color range calculation for anomalies/increments centered at zero
    if symmetric:
        abs_max = max(abs(np.nanmin(field)), abs(np.nanmax(field))) if (vmin is None or vmax is None) else max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    # Plot filled contour lines
    cf = ax.contourf(lon, lat, field, transform=ccrs.PlateCarree(),
                     levels=60, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')

    # Add Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08, shrink=0.75)
    cbar.set_label(colorbar_label, fontsize=4, fontweight='bold')

    plt.title(title, fontsize=8, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  -> Plot saved successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot 2D spatial maps of Analysis Increments and Errors using Cartopy.")
    # parser.add_argument("-i", "--input", required=True, help="Input reconstructed regular grid NetCDF dataset")
    # parser.add_argument("-r", "--ref", required=True, help="Reference dataset (e.g., GFS reference file)")
    parser.add_argument("-i", "--input", default="output/reconstructed_aida_analysis_20240106.t06z.1p00.nc", help="Input reconstructed regular grid NetCDF dataset")
    parser.add_argument("-r", "--ref", default="../data/regular_truth/gfs.20240106.t06z.1p00.f000.nc", help="Reference dataset (e.g., GFS reference file)")
    parser.add_argument("-v", "--var", default="t", help="Variable base name to analyze (default: t)")
    parser.add_argument("-l", "--level", type=int, default=10, help="Vertical level index to plot (default: 0)")
    parser.add_argument("-o", "--outdir", default="./plots", help="Directory destination to save output plots")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n========================================================")
    print(f"[AIDA PLOTTING] Generating Spatial Increments & Errors")
    print(f"Target Variable: {args.var} | Level Index: {args.level}")
    print(f"========================================================\n")

    ds = xr.open_dataset(args.input)
    ds_ref = xr.open_dataset(args.ref)

    lon = ds['lon'].values
    lat = ds['lat'].values

    # Determine variable naming strategy
    an_key = f"{args.var}_analysis" if f"{args.var}_analysis" in ds else args.var
    bg_key = f"{args.var}_background" if f"{args.var}_background" in ds else None
    inc_key = f"{args.var}_increment" if f"{args.var}_increment" in ds else None

    ref_key = args.var if args.var in ds_ref else f"{args.var}_analysis"

    # Extract 2D slices based on dimensionality (3D/4D vertical level selection)
    da_an = ds[an_key]
    if 'height' in da_an.dims:
        an_slice = da_an.isel(height=args.level).values
    elif 'level' in da_an.dims:
        an_slice = da_an.isel(level=args.level).values
    else:
        an_slice = da_an.values

    # Compute or extract Analysis Increment (Analysis - Background)
    if inc_key in ds:
        da_inc = ds[inc_key]
        inc_slice = da_inc.isel(height=args.level).values if 'height' in da_inc.dims else da_inc.values
    elif bg_key in ds:
        da_bg = ds[bg_key]
        bg_slice = da_bg.isel(height=args.level).values if 'height' in da_bg.dims else da_bg.values
        inc_slice = an_slice - bg_slice
    else:
        inc_slice = None

    # Compute Spatial Error (Analysis - Reference)
    da_ref = ds_ref[ref_key]
    if 'height' in da_ref.dims:
        ref_slice = da_ref.isel(height=args.level).values
    elif 'level' in da_ref.dims:
        ref_slice = da_ref.isel(level=args.level).values
    elif 'latitude' in da_ref.dims:  # Handle cases where GFS latitude/longitude ordering varies
        ref_slice = da_ref.isel(height=args.level).values if 'height' in da_ref.dims else da_ref.values
    else:
        ref_slice = da_ref.values

    # Ensure shape alignment between test and reference array
    if an_slice.shape != ref_slice.shape:
        print(f"Warning: Slices shape mismatch: test {an_slice.shape} vs ref {ref_slice.shape}. Squeezing arrays...")
        an_slice = np.squeeze(an_slice)
        ref_slice = np.squeeze(ref_slice)

    error_slice = an_slice - ref_slice

    # Get units from attributes
    units = da_an.attrs.get('units', '')
    unit_str = f" [{units}]" if units else ""

    # 1. Plot Analysis Field
    create_spatial_plot(
        lon, lat, an_slice,
        title=f"AIDA Analysis State ({args.var.upper()}) - Level Index {args.level}",
        colorbar_label=f"{args.var.upper()}{unit_str}",
        output_path=os.path.join(args.outdir, f"map_analysis_{args.var}_lvl{args.level}.png"),
        cmap='viridis'
    )

    # 2. Plot Analysis Increment (Analysis - Background)
    if inc_slice is not None:
        create_spatial_plot(
            lon, lat, inc_slice,
            title=f"AIDA Analysis Increment (Analysis - Background) - {args.var.upper()}",
            colorbar_label=f"Increment{unit_str}",
            output_path=os.path.join(args.outdir, f"map_increment_{args.var}_lvl{args.level}.png"),
            cmap='coolwarm',
            symmetric=True
        )

    # 3. Plot Absolute Error Distribution (Analysis - Reference)
    create_spatial_plot(
        lon, lat, error_slice,
        title=f"Analysis Spatial Error (Analysis - Reference) - {args.var.upper()}",
        colorbar_label=f"Error{unit_str}",
        output_path=os.path.join(args.outdir, f"map_error_{args.var}_lvl{args.level}.png"),
        cmap='bwr',
        symmetric=True
    )

    print("\n[COMPLETE] All spatial maps generated successfully!")


if __name__ == "__main__":
    main()

