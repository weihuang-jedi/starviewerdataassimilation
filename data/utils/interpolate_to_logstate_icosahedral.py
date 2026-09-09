#!/usr/bin/env python3
"""
interpolate_to_logstate_icosahedral.py
--------------------------------------
End-to-end processing pipeline that:
  1. Interpolates regular lat/lon NetCDF weather fields onto a 1D icosahedral grid layout.
  2. Directly converts thermodynamic fields (T, p) into log-state space (ln_t, ln_p, ln_rho).
  3. Computes air density (rho) via Ideal Gas Law if not provided.
  4. Formats CF-1.8/UGRID compliant output ready for AIDA GNN surrogate model ingestion.
"""

import argparse
import glob
import os
import warnings
import numpy as np
import xarray as xr

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
    Unified multi-level spatial interpolator and thermodynamic log-state transformation engine.
    """
    def __init__(self, master_mesh_path: str):
        self.master_mesh_path = master_mesh_path
        self.ds_mesh = None
        self._load_mesh()

    def _load_mesh(self) -> None:
        print(f"[AIDA PIPELINE] Loading master icosahedral mesh: {self.master_mesh_path}")
        self.ds_mesh = xr.open_dataset(self.master_mesh_path).load()

    def _apply_circular_padding(self, ds_src: xr.Dataset, lon_key: str) -> xr.Dataset:
        """Appends a 360-degree boundary column to eliminate prime-meridian seams."""
        raw_lons = ds_src[lon_key].values

        if raw_lons[-1] < 360.0:
            grid_res = raw_lons[1] - raw_lons[0]
            padded_lons = np.append(raw_lons, raw_lons[-1] + grid_res)

            padded_ds = xr.concat([ds_src, ds_src.isel({lon_key: slice(0, 1)})], dim=lon_key)
            padded_ds = padded_ds.assign_coords({lon_key: padded_lons})
            return padded_ds
        return ds_src

    def process_file(self, input_nc: str, output_nc: str) -> None:
        print(f"\n[AIDA PIPELINE] Processing: {input_nc}")
        ds_src = xr.open_dataset(input_nc)

        # 1. Identify target data payload variables
        coords_keys = {"time", "step", "valid_time", "height", "lat", "lon", "latitude", "longitude"}
        target_vars = [v for v in ds_src.data_vars if v not in coords_keys]

        if not target_vars:
            raise KeyError(f"Could not identify valid weather variables in: {input_nc}")

        # Standardize latitude ascending (-90 to +90)
        lat_key = 'lat' if 'lat' in ds_src.coords else 'latitude'
        if ds_src[lat_key].values[0] > ds_src[lat_key].values[-1]:
            ds_src = ds_src.sortby(lat_key)

        # Standardize longitude range [0, 360]
        lon_key = 'lon' if 'lon' in ds_src.coords else 'longitude'
        lon_vals = ds_src[lon_key].values
        if np.any(lon_vals < 0):
            ds_src = ds_src.assign_coords({lon_key: np.mod(lon_vals, 360.0)})
            ds_src = ds_src.sortby(lon_key)

        # 2. Apply Circular Longitude Padding
        padded_ds = self._apply_circular_padding(ds_src, lon_key)

        # 3. Horizontal Linear Interpolation onto Mesh Vertices
        mesh_lons = np.mod(self.ds_mesh['longitude'].values, 360.0)
        mesh_lats = self.ds_mesh['latitude'].values

        target_lon = xr.DataArray(mesh_lons, dims=["node"], coords={"node": self.ds_mesh.node})
        target_lat = xr.DataArray(mesh_lats, dims=["node"], coords={"node": self.ds_mesh.node})

        interpolated_fields = {}

        for var in target_vars:
            print(f"  -> Interpolating variable: '{var}'")
            cube = padded_ds[var].interp(
                {lon_key: target_lon, lat_key: target_lat},
                method="linear",
                kwargs={'bounds_error': False, 'fill_value': None}
            )

            var_data = cube.values

            # Enforce physical non-negativity constraint for humidity
            if var in ['q', 'qv', 'q_icosahedral']:
                var_data = np.clip(var_data, 0.0, None)

            # Standardize base output key name
            clean_var_name = var.replace("_icosahedral", "")
            interpolated_fields[clean_var_name] = (var_data, ds_src[var].attrs)

        # 4. Thermodynamic Transformation to Log-State Space
        print("  -> Performing thermodynamic transformation to log-state space (ln_t, ln_p, ln_rho)...")

        t_key = find_var(interpolated_fields.keys(), T_NAMES)
        p_key = find_var(interpolated_fields.keys(), P_NAMES)
        rho_key = find_var(interpolated_fields.keys(), RHO_NAMES)

        if not t_key or not p_key:
            raise ValueError(f"Missing required Temperature/Pressure keys in {input_nc}. Found: {list(interpolated_fields.keys())}")

        T_val = np.clip(interpolated_fields[t_key][0], 1e-5, None)
        p_val = np.clip(interpolated_fields[p_key][0], 1e-5, None)

        if rho_key:
            rho_val = np.clip(interpolated_fields[rho_key][0], 1e-8, None)
        else:
            # Ideal Gas Law derivation: rho = p / (R_d * T)
            rho_val = p_val / (R_D * T_val)

        ln_t_val = np.log(T_val)
        ln_p_val = np.log(p_val)
        ln_rho_val = np.log(rho_val)

        data_vars_out = {
            "ln_t_icosahedral": (["height", "node"], ln_t_val, {"long_name": "Logarithm of Temperature", "units": "ln(K)"}),
            "ln_p_icosahedral": (["height", "node"], ln_p_val, {"long_name": "Logarithm of Pressure", "units": "ln(Pa)"}),
            "ln_rho_icosahedral": (["height", "node"], ln_rho_val, {"long_name": "Logarithm of Density", "units": "ln(kg/m3)"})
        }

        # 5. Passthrough Non-Thermodynamic Variables (u, v, w, q, etc.)
        skip_keys = {t_key, p_key, rho_key} if rho_key else {t_key, p_key}
        for var_name, (data_arr, orig_attrs) in interpolated_fields.items():
            if var_name not in skip_keys:
                out_name = f"{var_name}_icosahedral" if not var_name.endswith("_icosahedral") else var_name
                new_attrs = orig_attrs.copy()
                new_attrs.update({
                    "coordinates": "longitude latitude",
                    "mesh": "icosahedral_mesh"
                })
                data_vars_out[out_name] = (["height", "node"], data_arr, new_attrs)

        # 6. Append Static Mesh Geometry
        for static_field in ['land_sea_mask', 'elevation']:
            if static_field in self.ds_mesh:
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

        # Coordinates definition
        coords_out = {
            "height": ds_src.height.values,
            "node": self.ds_mesh.node.values,
            "face": self.ds_mesh.face.values,
            "three": np.arange(3)
        }
        for time_coord in ['time', 'step', 'valid_time']:
            if time_coord in ds_src.coords:
                coords_out[time_coord] = ds_src[time_coord].values

        ds_output = xr.Dataset(
            data_vars=data_vars_out,
            coords=coords_out,
            attrs={
                "title": "AIDA GNN Log-State Weather Data on Global Icosahedral Grid",
                "source_dataset": ds_src.attrs.get("title", "Vertical Height Reanalysis File"),
                "conventions": "CF-1.8 UGRID-1.0"
            }
        )

        # Serialize directly
        os.makedirs(os.path.dirname(output_nc) or ".", exist_ok=True)
        ds_output.to_netcdf(output_nc, format="NETCDF4")

        ds_src.close()
        ds_output.close()
        print(f"  -> Successfully saved log-state dataset to: {output_nc}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Lat-Lon Interpolation and Log-State Transformation Pipeline")
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
