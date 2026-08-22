#!/usr/bin/env python3
"""
interpolate_to_logstate_icosahedral.py
--------------------------------------
End-to-end processing pipeline that:
  1. Interpolates regular lat/lon terrain-following NetCDF weather fields onto a 1D icosahedral grid layout.
  2. Eliminates Prime Meridian / Pacific seam stripes via circular longitude padding and SciPy RegularGridInterpolator.
  3. Re-orders descending latitudes to ascending (-90 to +90).
  4. Converts thermodynamic fields (T, p) into log-state space (ln_t, ln_p, ln_rho).
  5. Formats CF-1.8/UGRID compliant output ready for AIDA GNN surrogate model ingestion.
"""

import argparse
import glob
import os
import warnings
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Silence harmless warnings
warnings.filterwarnings("ignore")

R_D = 287.058  # Gas constant for dry air (J/(kg*K))

T_NAMES = ['t', 't_icosahedral', 'temperature', 'TMP', 'T']
P_NAMES = ['p', 'p_icosahedral', 'pressure', 'PRES', 'P']
RHO_NAMES = ['rho', 'rho_icosahedral', 'density', 'DEN']


def find_var(data_vars, name_list):
    """Utility to locate matching variable name from standard lookup aliases."""
    for name in name_list:
        if name in data_vars:
            return name
    return None


class IcosahedralLogStatePipeline:
    """
    Unified multi-level spatial interpolator for terrain-following grids onto icosahedral nodes
    and thermodynamic log-state transformation engine.
    """
    def __init__(self, master_mesh_path: str):
        self.master_mesh_path = master_mesh_path
        self.ds_mesh = None
        self._load_mesh()

    def _load_mesh(self) -> None:
        print(f"[AIDA PIPELINE] Loading master icosahedral mesh: {self.master_mesh_path}")
        self.ds_mesh = xr.open_dataset(self.master_mesh_path).load()

    def _prepare_source_dataset(self, ds_src: xr.Dataset, lat_key: str, lon_key: str) -> tuple[xr.Dataset, str, str]:
        """Ensures ascending lats/lons and appends 360-degree circular boundary column."""
        # 1. Reverse descending latitude (-90 to +90)
        if ds_src[lat_key].values[0] > ds_src[lat_key].values[-1]:
            print(f"  -> Reversing descending latitude coordinate '{lat_key}' to ascending [-90 to +90]")
            ds_src = ds_src.sortby(lat_key)

        # 2. Normalize longitude range [0, 360]
        lon_vals = ds_src[lon_key].values
        if np.any(lon_vals < 0.0):
            ds_src = ds_src.assign_coords({lon_key: np.mod(lon_vals, 360.0)})
        ds_src = ds_src.sortby(lon_key)

        # 3. Apply circular longitude padding at 360 deg to bridge 359 -> 360/0 meridian seam
        raw_lons = ds_src[lon_key].values
        if raw_lons[-1] < 360.0:
            grid_res = raw_lons[1] - raw_lons[0]
            padded_lons = np.append(raw_lons, raw_lons[-1] + grid_res)

            # Concatenate 0-degree slice onto the end as 360-degree slice
            padded_ds = xr.concat([ds_src, ds_src.isel({lon_key: slice(0, 1)})], dim=lon_key)
            padded_ds = padded_ds.assign_coords({lon_key: padded_lons})
            ds_src = padded_ds

        return ds_src

    def process_file(self, input_nc: str, output_nc: str) -> None:
        print(f"\n[AIDA PIPELINE] Processing Terrain-Following Grid: {input_nc}")
        ds_src = xr.open_dataset(input_nc)

        # Identify vertical level dimension ('level' or 'height')
        vert_dim = 'level' if 'level' in ds_src.dims else ('height' if 'height' in ds_src.dims else None)
        if vert_dim is None:
            for candidate in ['level', 'height', 'target_level']:
                if candidate in ds_src.coords or candidate in ds_src.dims:
                    vert_dim = candidate
                    break

        lat_key = 'lat' if 'lat' in ds_src.coords else 'latitude'
        lon_key = 'lon' if 'lon' in ds_src.coords else 'longitude'

        # 1. Identify target data payload variables
        coords_keys = {"time", "step", "valid_time", "level", "height", "target_level", "eta", "lat", "lon", "latitude", "longitude"}
        target_vars = [v for v in ds_src.data_vars if v not in coords_keys]

        if not target_vars:
            raise KeyError(f"Could not identify valid weather variables in: {input_nc}")

        # 2. Sanitize NaNs in input fields (e.g. w)
        for var in target_vars:
            if np.isnan(ds_src[var].values).any():
                print(f"  -> Filling NaNs in '{var}' with 0.0 prior to spatial interpolation")
                ds_src[var] = ds_src[var].fillna(0.0)

        # 3. Prepare dataset with circular longitude padding and ascending coordinates
        ds_src = self._prepare_source_dataset(ds_src, lat_key, lon_key)

        src_lats = ds_src[lat_key].values
        src_lons = ds_src[lon_key].values

        # Target icosahedral node positions
        mesh_lons = np.mod(self.ds_mesh['longitude'].values, 360.0)
        mesh_lats = np.clip(self.ds_mesh['latitude'].values, -89.99, 89.99)

        interpolated_fields = {}

        # 4. Perform Seam-Free Interpolation Variable-by-Variable
        for var in target_vars:
            print(f"  -> Interpolating field to icosahedral grid: '{var}'")
            var_data = ds_src[var].values  # Shape: [level, lat, lon] or [lat, lon]

            if var_data.ndim == 3:
                num_levels = var_data.shape[0]
                out_arr = np.zeros((num_levels, len(mesh_lats)), dtype=np.float32)

                for k in range(num_levels):
                    rgi = RegularGridInterpolator(
                        (src_lats, src_lons),
                        var_data[k],
                        method='linear',
                        bounds_error=False,
                        fill_value=None
                    )
                    out_arr[k] = rgi((mesh_lats, mesh_lons))
            elif var_data.ndim == 2:
                rgi = RegularGridInterpolator(
                    (src_lats, src_lons),
                    var_data,
                    method='linear',
                    bounds_error=False,
                    fill_value=None
                )
                out_arr = rgi((mesh_lats, mesh_lons))
            else:
                out_arr = var_data

            # Enforce non-negativity constraint for humidity
            if var in ['q', 'qv', 'q_icosahedral']:
                out_arr = np.clip(out_arr, 0.0, None)

            clean_var_name = var.replace("_icosahedral", "")
            interpolated_fields[clean_var_name] = (out_arr, ds_src[var].attrs)

        # 5. Thermodynamic Transformation to Log-State Space
        print("  -> Performing thermodynamic transformation to log-state space (ln_t, ln_p, ln_rho)...")

        t_key = find_var(interpolated_fields.keys(), T_NAMES)
        p_key = find_var(interpolated_fields.keys(), P_NAMES)
        rho_key = find_var(interpolated_fields.keys(), RHO_NAMES)

        if not t_key or not p_key:
            raise ValueError(f"Missing required Temperature/Pressure keys in {input_nc}. Found: {list(interpolated_fields.keys())}")

        T_val = np.clip(interpolated_fields[t_key][0], 180.0, 340.0)
        p_val = np.clip(interpolated_fields[p_key][0], 1.0, None)

        p_pa = p_val * 100.0 if np.nanmean(p_val) < 2000.0 else p_val

        if rho_key:
            rho_val = np.clip(interpolated_fields[rho_key][0], 1e-8, None)
        else:
            rho_val = p_pa / (R_D * T_val)

        ln_t_val = np.log(T_val)
        ln_p_val = np.log(p_pa)
        ln_rho_val = np.log(rho_val)

        level_dim_name = "level" if vert_dim else "level"
        num_levels = ln_t_val.shape[0] if ln_t_val.ndim == 2 else 32

        data_vars_out = {
            "ln_t_icosahedral": ([level_dim_name, "node"], ln_t_val, {"long_name": "Logarithm of Temperature", "units": "ln(K)"}),
            "ln_p_icosahedral": ([level_dim_name, "node"], ln_p_val, {"long_name": "Logarithm of Pressure", "units": "ln(Pa)"}),
            "ln_rho_icosahedral": ([level_dim_name, "node"], ln_rho_val, {"long_name": "Logarithm of Density", "units": "ln(kg/m3)"})
        }

        # 6. Passthrough Non-Thermodynamic Variables
        skip_keys = {t_key, p_key, rho_key} if rho_key else {t_key, p_key}
        for var_name, (data_arr, orig_attrs) in interpolated_fields.items():
            if var_name not in skip_keys:
                out_name = f"{var_name}_icosahedral" if not var_name.endswith("_icosahedral") else var_name
                new_attrs = orig_attrs.copy()
                new_attrs.update({
                    "coordinates": "longitude latitude",
                    "mesh": "icosahedral_mesh"
                })

                if data_arr.ndim == 2:
                    data_vars_out[out_name] = ([level_dim_name, "node"], data_arr, new_attrs)
                elif data_arr.ndim == 1:
                    data_vars_out[out_name] = (["node"], data_arr, new_attrs)

        # 7. Append Topography & Mesh Metadata
        if 'h_terrain' in interpolated_fields and 'h_terrain' not in data_vars_out:
            data_vars_out['h_terrain'] = (["node"], interpolated_fields['h_terrain'][0], ds_src['h_terrain'].attrs)

        for static_field in ['land_sea_mask', 'elevation']:
            if static_field in self.ds_mesh and static_field not in data_vars_out:
                data_vars_out[static_field] = (["node"], self.ds_mesh[static_field].values, self.ds_mesh[static_field].attrs)

        data_vars_out.update({
            "face_nodes": (["face", "three"], self.ds_mesh['face_nodes'].values, self.ds_mesh['face_nodes'].attrs),
            "longitude": (["node"], mesh_lons, {"units": "degrees_east", "standard_name": "longitude"}),
            "latitude": (["node"], mesh_lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "x_cartesian": (["node"], self.ds_mesh['x_cartesian'].values, {"units": "m"}),
            "y_cartesian": (["node"], self.ds_mesh['y_cartesian'].values, {"units": "m"}),
            "z_cartesian": (["node"], self.ds_mesh['z_cartesian'].values, {"units": "m"}),
            "icosahedral_mesh": ([], 0, self.ds_mesh['icosahedral_mesh'].attrs)
        })

        coords_out = {
            level_dim_name: np.arange(1, num_levels + 1, dtype=np.int32),
            "node": self.ds_mesh.node.values,
            "face": self.ds_mesh.face.values,
            "three": np.arange(3)
        }

        if "eta" in ds_src.coords:
            coords_out["eta"] = (level_dim_name, ds_src["eta"].values, ds_src["eta"].attrs)
        if "target_level" in ds_src.coords:
            coords_out["target_level"] = (level_dim_name, ds_src["target_level"].values, ds_src["target_level"].attrs)

        for time_coord in ['time', 'step', 'valid_time']:
            if time_coord in ds_src.coords:
                coords_out[time_coord] = ds_src[time_coord].values

        ds_output = xr.Dataset(
            data_vars=data_vars_out,
            coords=coords_out,
            attrs={
                "title": "AIDA GNN Log-State Weather Data on Global Icosahedral Grid with Terrain Following",
                "source_dataset": ds_src.attrs.get("title", "Terrain-Following Reanalysis File"),
                "terrain_formula": ds_src.attrs.get("terrain_formula", "h = Hmax - eta * (Hmax - Hterrain)"),
                "conventions": "CF-1.8 UGRID-1.0"
            }
        )

        os.makedirs(os.path.dirname(output_nc) or ".", exist_ok=True)
        ds_output.to_netcdf(output_nc, format="NETCDF4")

        ds_src.close()
        ds_output.close()
        print(f"  -> Successfully saved clean log-state dataset to: {output_nc}")


def main():
    parser = argparse.ArgumentParser(description="Terrain-Following Lat-Lon to Icosahedral Log-State Pipeline")
    parser.add_argument("-i", "--input", help="Single input regular lat-lon NetCDF file")
    parser.add_argument("--input_dir", help="Directory containing regular lat-lon NetCDF files")
    parser.add_argument("-m", "--mesh", required=True, help="Path to master icosahedral mesh file")
    parser.add_argument("-o", "--output", help="Destination path for single output file")
    parser.add_argument("--output_dir", help="Destination directory for batch output files")
    parser.add_argument("--pattern", default="*.nc", help="Pattern to match files in --input_dir")

    args = parser.parse_args()

    pipeline = IcosahedralLogStatePipeline(master_mesh_path=args.mesh)

    if args.input and args.output:
        pipeline.process_file(args.input, args.output)
    elif args.input_dir and args.output_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not files:
            print(f"[ERROR] No files matching '{args.pattern}' found in {args.input_dir}")
            return
        for f in files:
            out_path = os.path.join(args.output_dir, os.path.basename(f))
            pipeline.process_file(f, out_path)
    else:
        parser.error("Must provide either (-i/--input and -o/--output) or (--input_dir and --output_dir)")


if __name__ == "__main__":
    main()
