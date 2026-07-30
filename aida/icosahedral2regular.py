#!/usr/bin/env python3
import os
import argparse
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def regrid_dataset(input_file: str, output_file: str, grid_file: str = None, resolution: float = 1.0):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Opening data dataset: {input_file}")
    ds = xr.open_dataset(input_file)

    # 1. Obtain coordinates from external grid file or input dataset
    if grid_file:
        if not os.path.exists(grid_file):
            raise FileNotFoundError(f"Specified grid file not found: {grid_file}")
        print(f"Loading grid coordinates from external file: {grid_file}")
        ds_grid = xr.open_dataset(grid_file)

        lons_raw = None
        lats_raw = None
        for lon_key in ['longitude', 'lon', 'clon']:
            if lon_key in ds_grid:
                lons_raw = ds_grid[lon_key].values
                break
        for lat_key in ['latitude', 'lat', 'clat']:
            if lat_key in ds_grid:
                lats_raw = ds_grid[lat_key].values
                break

        if lons_raw is None or lats_raw is None:
            raise KeyError("Could not find longitude or latitude in the provided grid file.")

        if lons_raw.ndim == 1 and lats_raw.ndim == 1 and lons_raw.shape[0] != lats_raw.shape[0]:
            print(f"Detected 1D coordinate axes: lons {lons_raw.shape}, lats {lats_raw.shape}. Expanding to 2D meshgrid.")
            mesh_lon, mesh_lat = np.meshgrid(lons_raw, lats_raw)
            src_lons = mesh_lon.flatten()
            src_lats = mesh_lat.flatten()
        else:
            src_lons = np.squeeze(lons_raw)
            src_lats = np.squeeze(lats_raw)
    else:
        src_lons = np.squeeze(ds['longitude'].values)
        src_lats = np.squeeze(ds['latitude'].values)

    # Check for unit conversion (radians to degrees)
    if np.nanmax(src_lons) <= 2 * np.pi and np.nanmax(src_lats) <= np.pi:
        print("Converting coordinate units from radians to degrees...")
        src_lons = np.degrees(src_lons)
        src_lats = np.degrees(src_lats)

    # Wrap negative longitudes [-180..180] to [0..360]
    src_lons = np.where(src_lons < 0, src_lons + 360, src_lons)

    # Mask single-point NaNs safely
    valid_coord_mask = ~np.isnan(src_lons) & ~np.isnan(src_lats)
    src_lons_clean = src_lons[valid_coord_mask].astype(np.float64)
    src_lats_clean = src_lats[valid_coord_mask].astype(np.float64)
    points = np.vstack((src_lons_clean, src_lats_clean)).T

    print(f"Successfully loaded {len(points)} valid source coordinate points.")

    # 2. Setup target structured output grid
    grid_lons = np.arange(0.0, 360.0, resolution, dtype=np.float64)
    grid_lats = np.arange(-90.0, 90.0 + resolution, resolution, dtype=np.float64)
    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

    print("Building spatial triangulation interpolators...")
    dummy_values = np.zeros(len(points), dtype=np.float64)
    linear_interp = LinearNDInterpolator(points, dummy_values)
    nearest_interp = NearestNDInterpolator(points, dummy_values)

    # Target variables containing spatial node dimension
    node_vars = [v for v in ds.data_vars if 'node' in ds[v].dims]
    print(f"Variables target for regridding: {node_vars}")

    regrid_dict = {}

    for var_name in node_vars:
        da = ds[var_name]
        clean_name = var_name.replace('_icosahedral', '') if var_name.endswith('_icosahedral') else var_name
        print(f"\nProcessing variable: {var_name} -> saving as: {clean_name}...")

        has_time = 'time' in da.dims
        has_height = 'height' in da.dims

        time_len = ds['time'].size if 'time' in ds else 1
        height_len = len(ds['height']) if has_height else 1

        def interpolate_layer(layer_raw):
            layer_flat = np.squeeze(layer_raw).flatten()[valid_coord_mask].astype(np.float64)

            # Execute pre-calculated Delaunay triangulation
            linear_interp.values = layer_flat.reshape(-1, 1)
            grid_data = linear_interp(lon_mesh, lat_mesh)

            # Fill convex hull boundary gaps using nearest neighbor
            if np.isnan(grid_data).any():
                nearest_interp.values = layer_flat.reshape(-1, 1)
                nearest_evaluated = nearest_interp(lon_mesh, lat_mesh)

                # Flatten arrays to 1D to guarantee safe 1D boolean indexing
                grid_flat = grid_data.ravel()
                nan_mask_flat = np.isnan(grid_flat)
                nearest_flat = nearest_evaluated.ravel()

                grid_flat[nan_mask_flat] = nearest_flat[nan_mask_flat]
                grid_data = grid_flat.reshape(lon_mesh.shape)

            return grid_data.astype(np.float32)

        if has_time and has_height:
            time_stack = []
            for t_idx in range(time_len):
                height_stack = []
                for h_idx in range(height_len):
                    print(f"  -> Time {t_idx+1}/{time_len} | Height {h_idx+1}/{height_len}", end="\r")
                    layer_data = da.isel(time=t_idx, height=h_idx).values
                    height_stack.append(interpolate_layer(layer_data))
                time_stack.append(np.stack(height_stack, axis=0))

            print(f"\n  -> Finished 4D reconstruction of {clean_name}.")
            regrid_array = np.stack(time_stack, axis=0)
            dims = ('time', 'height', 'lat', 'lon')

        elif has_height:
            height_stack = []
            for h_idx in range(height_len):
                print(f"  -> Height {h_idx+1}/{height_len}", end="\r")
                layer_data = da.isel(height=h_idx).values
                height_stack.append(interpolate_layer(layer_data))

            print(f"\n  -> Finished 3D reconstruction of {clean_name}.")
            regrid_array = np.stack(height_stack, axis=0)
            dims = ('height', 'lat', 'lon')

        else:
            print(f"  -> Processing 2D surface layer")
            regrid_array = interpolate_layer(da.values)
            dims = ('lat', 'lon')

        regrid_dict[clean_name] = xr.DataArray(
            data=regrid_array,
            dims=dims,
            attrs=da.attrs
        )

    # Construct coordinates safely
    coords = {
        'lon': ('lon', grid_lons.astype(np.float32), {'units': 'degrees_east', 'standard_name': 'longitude'}),
        'lat': ('lat', grid_lats.astype(np.float32), {'units': 'degrees_north', 'standard_name': 'latitude'})
    }

    if 'height' in ds.coords:
        coords['height'] = ('height', ds['height'].values, ds['height'].attrs)

    # Safely handle scalar or 1D time coordinates
    if 'time' in ds.coords:
        t_val = ds['time'].values
        if t_val.ndim == 0:
            # Scalar time coordinate -> store as scalar without forcing dimension name
            coords['time'] = xr.Variable((), t_val, attrs=ds['time'].attrs)
        else:
            coords['time'] = ('time', t_val, ds['time'].attrs)

    output_ds = xr.Dataset(data_vars=regrid_dict, coords=coords, attrs=ds.attrs)

    print(f"\nSaving structured regular grid NetCDF to: {output_file}")
    output_ds.to_netcdf(output_file, format="NETCDF4")
    print("Success! Interpolation complete.")

def main():
    parser = argparse.ArgumentParser(description="Regrid unstructured icosahedral NetCDF data to regular lat-lon.")
    parser.add_argument("-i", "--input", required=True, help="Input unstructured NetCDF file")
    parser.add_argument("-o", "--output", required=True, help="Output regular lat-lon NetCDF destination")
    parser.add_argument("-g", "--grid", required=False, help="Path to external grid NetCDF file containing longitude/latitude")
    parser.add_argument("-r", "--res", type=float, default=1.0, help="Grid resolution (default: 1.0)")

    args = parser.parse_args()
    regrid_dataset(args.input, args.output, args.grid, args.res)

if __name__ == "__main__":
    main()
