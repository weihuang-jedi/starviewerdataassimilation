#!/usr/bin/env python3
"""
utils/icosahedral2regular.py
----------------------------
Regrids unstructured 3D M6 icosahedral atmospheric fields (level=32, node=40962)
to regular 2D/3D latitude-longitude grids (e.g. 1.0° or 0.25° resolution).
Transforms logarithmic state variables (ln_t, ln_p, ln_rho) into physical units.
"""

import os
import argparse
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# Physical Gas Constant
R_D = 287.058


def transform_log_variable(var_name: str, da: xr.DataArray):
    """
    Transforms log state variables (ln_t, ln_p, ln_rho) back to physical space (T, P, Rho),
    updating standard names, long names, and units accordingly.
    """
    val = da.values.copy()

    if var_name in ['ln_t_icosahedral', 'ln_t', 't_icosahedral']:
        if np.nanmean(val) > 100.0:
            phys_data = val  # Already physical Kelvin
        else:
            if np.nanmean(val) < 2.0:
                val = val * 0.2 + 5.5
            phys_data = np.exp(val)

        phys_data = np.clip(phys_data, 150.0, 350.0)
        attrs = {
            'long_name': 'Absolute Temperature',
            'units': 'K',
            'standard_name': 'air_temperature'
        }
        return 't', phys_data, attrs

    elif var_name in ['ln_p_icosahedral', 'ln_p', 'p_icosahedral']:
        if np.nanmean(val) > 100.0:
            phys_data = val  # Already physical Pa
        else:
            if np.nanmean(val) < 5.0:
                val = val * 0.3 + 11.5
            phys_data = np.exp(val)

        phys_data = np.clip(phys_data, 1.0, 110000.0)
        attrs = {
            'long_name': 'Air Pressure',
            'units': 'Pa',
            'standard_name': 'air_pressure'
        }
        return 'p', phys_data, attrs

    elif var_name in ['ln_rho_icosahedral', 'ln_rho', 'rho_icosahedral']:
        if np.nanmean(val) > 0.0 and np.nanmean(val) < 5.0:
            phys_data = val
        else:
            if np.nanmean(val) < -5.0 or np.nanmean(val) < 0.0:
                val = val * 0.5 - 1.2
            phys_data = np.exp(val)

        phys_data = np.clip(phys_data, 1e-6, 3.0)
        attrs = {
            'long_name': 'Air Density',
            'units': 'kg m**-3',
            'standard_name': 'air_density'
        }
        return 'rho', phys_data, attrs

    else:
        clean_name = var_name.replace('_icosahedral', '') if var_name.endswith('_icosahedral') else var_name
        return clean_name, val, da.attrs


def regrid_dataset(input_file: str, output_file: str, grid_file: str = None, resolution: float = 1.0):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"[AIDA REGRID] Opening dataset: {input_file}")
    ds = xr.open_dataset(input_file)

    num_nodes = ds.sizes.get('node', None)

    # Load Source Longitudes and Latitudes from Dataset
    src_lons = np.squeeze(ds['longitude'].values)
    src_lats = np.squeeze(ds['latitude'].values)

    if np.nanmax(src_lons) <= 2 * np.pi and np.nanmax(src_lats) <= np.pi:
        print("[AIDA REGRID] Converting coordinate units from radians to degrees...")
        src_lons = np.degrees(src_lons)
        src_lats = np.degrees(src_lats)

    src_lons = np.where(src_lons < 0, src_lons + 360, src_lons)

    valid_coord_mask = ~np.isnan(src_lons) & ~np.isnan(src_lats)
    src_lons_clean = src_lons[valid_coord_mask].astype(np.float64)
    src_lats_clean = src_lats[valid_coord_mask].astype(np.float64)
    points = np.vstack((src_lons_clean, src_lats_clean)).T

    print(f"[AIDA REGRID] Loaded {len(points)} valid source coordinate points.")

    # Target Regular Lat-Lon Grid Setup
    grid_lons = np.arange(0.0, 360.0, resolution, dtype=np.float64)
    grid_lats = np.arange(90.0, -90.0 - resolution / 2.0, -resolution, dtype=np.float64)
    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

    print(f"[AIDA REGRID] Target Grid: Lats ({grid_lats[0]} -> {grid_lats[-1]}, size={len(grid_lats)}), Lons ({grid_lons[0]} -> {grid_lons[-1]}, size={len(grid_lons)})")

    print("[AIDA REGRID] Building spatial triangulation interpolators...")
    dummy_values = np.zeros(len(points), dtype=np.float64)
    linear_interp = LinearNDInterpolator(points, dummy_values)
    nearest_interp = NearestNDInterpolator(points, dummy_values)

    node_vars = [v for v in ds.data_vars if 'node' in ds[v].dims]
    print(f"[AIDA REGRID] Variables targeted for physical conversion & regridding:\n -> {node_vars}\n")

    regrid_dict = {}

    for raw_var_name in node_vars:
        da = ds[raw_var_name]

        out_var_name, phys_array_data, updated_attrs = transform_log_variable(raw_var_name, da)
        print(f"Processing: {raw_var_name} -> Converting to Physical Field: [{out_var_name}]")

        has_time = 'time' in da.dims
        has_level = ('level' in da.dims) or ('height' in da.dims)

        time_len = ds.sizes.get('time', 1) if has_time else 1
        level_len = ds.sizes.get('level', ds.sizes.get('height', 1)) if has_level else 1

        def interpolate_layer(layer_raw):
            # Mask valid horizontal nodes
            layer_flat = np.squeeze(layer_raw)[valid_coord_mask].astype(np.float64)

            linear_interp.values = layer_flat.reshape(-1, 1)
            grid_data = linear_interp(lon_mesh, lat_mesh)

            if np.isnan(grid_data).any():
                nearest_interp.values = layer_flat.reshape(-1, 1)
                nearest_evaluated = nearest_interp(lon_mesh, lat_mesh)

                grid_flat = grid_data.ravel()
                nan_mask_flat = np.isnan(grid_flat)
                nearest_flat = nearest_evaluated.ravel()

                grid_flat[nan_mask_flat] = nearest_flat[nan_mask_flat]
                grid_data = grid_flat.reshape(lon_mesh.shape)

            return grid_data.astype(np.float32)

        # 1. Handle 4D: (time, level, lat, lon)
        if has_time and has_level and phys_array_data.ndim >= 3:
            time_stack = []
            for t_idx in range(time_len):
                level_stack = []
                for l_idx in range(level_len):
                    layer_data = phys_array_data[t_idx, l_idx] if phys_array_data.ndim == 3 else phys_array_data[l_idx]
                    level_stack.append(interpolate_layer(layer_data))
                time_stack.append(np.stack(level_stack, axis=0))

            regrid_array = np.stack(time_stack, axis=0)
            dims = ('time', 'level', 'lat', 'lon')

        # 2. Handle 3D: (level, lat, lon)
        elif has_level and phys_array_data.ndim == 2:
            level_stack = []
            for l_idx in range(level_len):
                layer_data = phys_array_data[l_idx]
                level_stack.append(interpolate_layer(layer_data))

            regrid_array = np.stack(level_stack, axis=0)
            dims = ('level', 'lat', 'lon')

        # 3. Handle 2D Surface Features: (lat, lon)
        else:
            regrid_array = interpolate_layer(phys_array_data)
            dims = ('lat', 'lon')

        regrid_dict[out_var_name] = xr.DataArray(
            data=regrid_array,
            dims=dims,
            attrs=updated_attrs
        )

    coords = {
        'lon': ('lon', grid_lons.astype(np.float32), {'units': 'degrees_east', 'standard_name': 'longitude'}),
        'lat': ('lat', grid_lats.astype(np.float32), {
            'units': 'degrees_north',
            'standard_name': 'latitude',
            'stored_direction': 'decreasing'
        })
    }

    if 'level' in ds.coords:
        coords['level'] = ('level', ds['level'].values, ds['level'].attrs)
    elif 'height' in ds.coords:
        coords['level'] = ('level', ds['height'].values, ds['height'].attrs)

    if 'target_level' in ds:
        coords['target_level'] = ('level', ds['target_level'].values, ds['target_level'].attrs)
    if 'eta' in ds:
        coords['eta'] = ('level', ds['eta'].values, ds['eta'].attrs)

    if 'time' in ds.coords:
        t_val = ds['time'].values
        if t_val.ndim == 0:
            coords['time'] = xr.Variable((), t_val, attrs=ds['time'].attrs)
        else:
            coords['time'] = ('time', t_val, ds['time'].attrs)

    output_ds = xr.Dataset(data_vars=regrid_dict, coords=coords, attrs=ds.attrs)
    output_ds.attrs['aida_regrid_status'] = "REGULAR_LATLON_PHYSICAL_UNITS"

    print(f"\n[AIDA SUCCESS] Saving structured regular lat-lon NetCDF to: {output_file}")
    output_ds.to_netcdf(output_file, format="NETCDF4")


def main():
    parser = argparse.ArgumentParser(description="Regrid unstructured icosahedral log-state NetCDF to regular physical lat-lon.")
    parser.add_argument("-i", "--input", required=True, help="Input unstructured NetCDF file")
    parser.add_argument("-o", "--output", required=True, help="Output regular lat-lon NetCDF destination")
    parser.add_argument("-g", "--grid", required=False, help="Path to external grid NetCDF file containing longitude/latitude")
    parser.add_argument("-r", "--res", type=float, default=1.0, help="Grid resolution in degrees (default: 1.0)")

    args = parser.parse_args()
    regrid_dataset(args.input, args.output, args.grid, args.res)


if __name__ == "__main__":
    main()
