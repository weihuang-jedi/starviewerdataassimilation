#!/usr/bin/env python
import os
import argparse
import warnings
import numpy as np
import xarray as xr
import cfgrib
from scipy.interpolate import interp1d

# Completely silence all library deprecation, future, and runtime warnings
warnings.filterwarnings("ignore")
os.environ["GRIB_API_LOG_LEVEL"] = "0"

class VerticalHeightInterpolator:
    """
    Interpolates standard isobaric pressure-level weather fields into a clean
    constant geometric height coordinate framework (Z) using vectorized mapping.
    Adapts dynamically to whatever subset of core fields are present in the GRIB file.
    Extrapolates out-of-bounds coordinates linearly to ensure suitability for Machine Learning.
    Ensures physically invalid negative values for specific humidity (q) are clipped.
    """
    def __init__(self, input_path: str, output_path: str):
        self.input_path = os.path.abspath(input_path)
        self.output_path = os.path.abspath(output_path)

        self.target_levels = np.array([
            2, 10, 20, 50, 75, 100, 150, 200, 300, 400,
            500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000,
            4500, 5000, 6000, 7000, 8000, 9000, 10000, 11500, 13000, 15000,
            17500, 20000
        ], dtype=np.float32)

    def process_and_interpolate(self):
        print(f"[LOAD] Extracting source fields from file: {os.path.basename(self.input_path)}")

        datasets = cfgrib.open_datasets(self.input_path, backend_kwargs={'errors': 'ignore'})

        # Gather and merge all datasets that contain 'isobaricInhPa' coordinates
        isobaric_parts = [ds for ds in datasets if 'isobaricInhPa' in ds.coords]
        
        if not isobaric_parts:
            raise KeyError("Could not locate any 'isobaricInhPa' pressure levels inside this file.")
        
        # Merge compatible parts together to prevent variables from being left out
        isobaric_ds = xr.merge(isobaric_parts, compat='override')

        available_vars = set(isobaric_ds.data_vars.keys())
        var_map = {}

        # Core mappings for available fields
        if 't' in available_vars: var_map['t'] = 't'
        elif 'tmp' in available_vars: var_map['t'] = 'tmp'

        if 'u' in available_vars: var_map['u'] = 'u'
        if 'v' in available_vars: var_map['v'] = 'v'

        if 'w' in available_vars: var_map['w'] = 'w'
        elif 'vvel' in available_vars: var_map['w'] = 'vvel'
        elif 'w_isobaric' in available_vars: var_map['w'] = 'w_isobaric'

        # Ensure 'r' is mapped if 'q' is not natively found in the 3D pressure levels
        if 'q' in available_vars: var_map['q'] = 'q'
        elif 'shum' in available_vars: var_map['q'] = 'shum'
        elif 'q_isobaric' in available_vars: var_map['q'] = 'q_isobaric'
        elif 'r' in available_vars: var_map['r'] = 'r'  # Map relative humidity as a fallback source

        if 'gh' in available_vars: var_map['gh'] = 'gh'

        # Geopotential Height (gh) is mandatory as our reference vertical coordinate
        if 'gh' not in var_map:
            raise KeyError(f"Geopotential Height ('gh') missing. Cannot establish vertical coordinate profiles. Available: {available_vars}")

        # --- NEW SECTION: DYNAMICALLY DERIVE 3D SPECIFIC HUMIDITY FROM RELATIVE HUMIDITY ---
        if 'q' not in var_map and 'r' in var_map:
            print(" -> 'q' missing on pressure levels. Natively calculating 3D Specific Humidity from Relative Humidity ('r')...")
            
            # Extract underlying numpy arrays for thermodynamics
            t_k = isobaric_ds[var_map['t']].values  # Temperature in Kelvin
            r_pct = isobaric_ds[var_map['r']].values  # Relative Humidity in %
            
            # Replicate pressure vector across spatial coordinates to match shapes (levs, lats, lons)
            p_v = isobaric_ds['isobaricInhPa'].values
            p_hpa = p_v[:, np.newaxis, np.newaxis] 

            # Calculate Saturation Vapor Pressure (hPa) using Tetens' formula
            t_c = t_k - 273.15
            e_s = 6.112 * np.exp((17.67 * t_c) / (t_k - 29.65))
            
            # Actual vapor pressure
            e = (r_pct / 100.0) * e_s
            
            # Calculate Specific Humidity (kg/kg)
            q_values = (0.622 * e) / (p_hpa - (0.378 * e))
            
            # Enforce strict physical limit boundary constraint
            q_values = np.clip(q_values, 1.0e-7, None)

            # Assign back into our dataset container as a standard 3D variable
            isobaric_ds['q'] = (isobaric_ds[var_map['t']].dims, q_values.astype(np.float32))
            isobaric_ds['q'].attrs = {
                "GRIB_shortName": "q",
                "units": "kg kg**-1",
                "long_name": "Specific Humidity Derived from Relative Humidity"
            }
            var_map['q'] = 'q'  # Register it so the rest of your pipeline processes it seamlessly
        # ----------------------------------------------------------------------------------

        print(" -> Resolved Variable Mapping Identifiers Found:")
        for k, v in var_map.items():
            if k in ['t', 'u', 'v', 'w', 'q', 'gh']:  # Filter out 'r' if we converted it
                print(f"    * Field '{k}' -> extracting from file key '{v}'")

        gh_var = isobaric_ds[var_map['gh']]
        z_src = gh_var.values

        print(f" -> Mapping target vertical dimensions: {len(self.target_levels)} height levels.")

        pressures = isobaric_ds['isobaricInhPa'].values
        lats = isobaric_ds['latitude'].values
        lons = isobaric_ds['longitude'].values
        has_time = 'time' in isobaric_ds.coords

        n_levs, n_lats, n_lons = z_src.shape[-3], z_src.shape[-2], z_src.shape[-1]
        out_shape = (len(self.target_levels), n_lats, n_lons)

        # Build storage arrays only for variables that are actually present
        active_keys = [k for k in ['t', 'u', 'v', 'w', 'q'] if k in var_map]

        interpolated_fields = {k: np.full(out_shape, np.nan, dtype=np.float32) for k in active_keys}
        interpolated_fields['p'] = np.full(out_shape, np.nan, dtype=np.float32)

        print("[INTERP] Running vertical coordinate interpolation over spatial grid nodes...")

        for lat_idx in range(n_lats):
            for lon_idx in range(n_lons):
                z_column = z_src[:, lat_idx, lon_idx]
                sort_idx = np.argsort(z_column)
                z_sorted = z_column[sort_idx]

                # Linearly interpolate/extrapolate available fields onto target heights
                for target_key in active_keys:
                    file_key = var_map[target_key]
                    var_column = isobaric_ds[file_key].values[:, lat_idx, lon_idx][sort_idx]
                    
                    interp_func = interp1d(z_sorted, var_column, kind='linear', fill_value='extrapolate')
                    val_column = interp_func(self.target_levels)
                    
                    # Prevent specific humidity from going negative during extrapolation
                    if target_key == 'q':
                        val_column = np.clip(val_column, 1.0e-7, None)
                       #val_column = np.clip(val_column, 0.0, None)
                        
                    interpolated_fields[target_key][:, lat_idx, lon_idx] = val_column

                # Save the ambient pressure corresponding to this height position (with extrapolation)
                p_column = pressures[sort_idx]
                interp_func_p = interp1d(z_sorted, p_column, kind='linear', fill_value='extrapolate')
                interpolated_fields['p'][:, lat_idx, lon_idx] = interp_func_p(self.target_levels)

        print("[PACKAGE] Constructing clean NetCDF4 dataset container...")
        coords = {
            "height": ("height", self.target_levels, {"units": "meters", "long_name": "Geometric Height Above Sea Level"}),
            "latitude": ("latitude", lats, isobaric_ds['latitude'].attrs),
            "longitude": ("longitude", lons, isobaric_ds['longitude'].attrs),
        }

        if has_time:
            coords["time"] = isobaric_ds['time']

        data_vars = {}
        for target_key in active_keys:
            file_key = var_map[target_key]
            data_vars[target_key] = (["height", "latitude", "longitude"], interpolated_fields[target_key], isobaric_ds[file_key].attrs)

        data_vars['p'] = (
            ["height", "latitude", "longitude"],
            interpolated_fields['p'],
            {"units": "hPa", "long_name": "Atmospheric Pressure at Target Height"}
        )

        ds_out = xr.Dataset(data_vars=data_vars, coords=coords, attrs={"title": "GFS Variables Interpolated to Constant Height Levels"})

        print(f"[SAVE] Serializing final netCDF structure to destination: {self.output_path}")
        ds_out.to_netcdf(self.output_path, format="NETCDF4")

        isobaric_ds.close()
        for ds in datasets:
            ds.close()
        ds_out.close()
        print("SUCCESS: Vertical transformation finalized completely.\n")

def main():
    parser = argparse.ArgumentParser(description="Vertically re-map GFS datasets from pressure coordinates to geometric heights.")
    parser.add_argument("-i", "--input", required=True, help="Path to input GRIB2 file")
    parser.add_argument("-o", "--output", required=True, help="Path to output NetCDF4 file")
    args = parser.parse_args()

    interpolator = VerticalHeightInterpolator(input_path=args.input, output_path=args.output)
    interpolator.process_and_interpolate()

if __name__ == "__main__":
    main()
