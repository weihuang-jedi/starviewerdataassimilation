#!/usr/bin/env python
import argparse
import os
import xarray as xr
import numpy as np
import warnings

# Silence harmless warnings
warnings.filterwarnings("ignore")

class LatLonToIcosahedralInterpolator:
    """
    An object-oriented multi-level spatial processing engine designed to interpolate
    regular lat/lon weather fields from height levels onto a 1D unstructured global
    icosahedral grid topology layout (UGRID/GraphCast mesh structure).
    """
    def __init__(self, input_weather_path: str, output_path: str, master_mesh_path: str):
        self.input_weather_path = input_weather_path
        self.output_path = output_path
        self.master_mesh_path = master_mesh_path

        self.ds_mesh = None
        self.ds_src = None
        self.target_vars = []

    def load_and_align_structures(self) -> None:
        """Loads grid definitions and opens height profile datasets."""
        print(f"[STAGE 1] Loading master icosahedral mesh definitions: {self.master_mesh_path}")
        self.ds_mesh = xr.open_dataset(self.master_mesh_path).load()

        print(f"[STAGE 2] Opening regular lat-lon height dataset: {self.input_weather_path}")
        self.ds_src = xr.open_dataset(self.input_weather_path)

        # Dynamically discover all weather variables mapped to (height, latitude, longitude)
        coords_keys = {"time", "step", "valid_time", "height", "lat", "lon", "latitude", "longitude"}
        self.target_vars = [v for v in self.ds_src.data_vars if v not in coords_keys]

        if not self.target_vars:
            raise KeyError(f"Could not identify valid weather variables in: {self.input_weather_path}")

        print(f" -> Automatically detected target data payload keys: {self.target_vars}")

        # Standardize latitude direction to ascending (-90 to +90)
        lat_key = 'lat' if 'lat' in self.ds_src.coords else 'latitude'
        if self.ds_src[lat_key].values[0] > self.ds_src[lat_key].values[-1]:
            print(" -> Reorienting regular latitude grid axis to strict ascending layout (-90 to +90)...")
            self.ds_src = self.ds_src.sortby(lat_key)

        # Standardize longitude range to [0, 360]
        lon_key = 'lon' if 'lon' in self.ds_src.coords else 'longitude'
        lon_vals = self.ds_src[lon_key].values
        if np.any(lon_vals < 0):
            print(" -> Standardizing input longitude coordinate axis from [-180, 180] to [0, 360]...")
            self.ds_src = self.ds_src.assign_coords({lon_key: np.mod(lon_vals, 360.0)})
            self.ds_src = self.ds_src.sortby(lon_key)

    def apply_circular_padding(self) -> xr.Dataset:
        """Appends a 360-degree boundary column to eliminate prime-meridian interpolation seams."""
        print("[STAGE 3] Computing circular longitude wrapping tensors...")
        lon_key = 'lon' if 'lon' in self.ds_src.coords else 'longitude'
        raw_lons = self.ds_src[lon_key].values

        # Append first slice (0 deg) to the end as 360 deg if not already present
        if raw_lons[-1] < 360.0:
            grid_res = raw_lons[1] - raw_lons[0]
            padded_lons = np.append(raw_lons, raw_lons[-1] + grid_res)
            
            padded_ds = xr.concat([self.ds_src, self.ds_src.isel({lon_key: slice(0, 1)})], dim=lon_key)
            padded_ds = padded_ds.assign_coords({lon_key: padded_lons})
            return padded_ds
        return self.ds_src

    def run_interpolation_pipeline(self) -> None:
        """Executes vectorized horizontal linear interpolation over the icosahedral grid vertices."""
        self.load_and_align_structures()
        padded_ds = self.apply_circular_padding()

        # Isolate spatial tracking coordinates from icosahedral mesh vertices
        mesh_lons = np.mod(self.ds_mesh['longitude'].values, 360.0)
        mesh_lats = self.ds_mesh['latitude'].values

        print(f"[STAGE 4] Executing spatial interpolation onto icosahedral grid nodes...")
        target_lon = xr.DataArray(mesh_lons, dims=["node"], coords={"node": self.ds_mesh.node})
        target_lat = xr.DataArray(mesh_lats, dims=["node"], coords={"node": self.ds_mesh.node})

        lon_key = 'lon' if 'lon' in padded_ds.coords else 'longitude'
        lat_key = 'lat' if 'lat' in padded_ds.coords else 'latitude'

        data_vars_out = {}

        for var in self.target_vars:
            print(f" -> Interpolating variable: '{var}'")
            interpolated_cube = padded_ds[var].interp(
                {lon_key: target_lon, lat_key: target_lat},
                method="linear",
                kwargs={'bounds_error': False, 'fill_value': None}
            )

            var_data = interpolated_cube.values

            # Keep physical non-negativity constraint for specific humidity
            if var in ['q', 'qv', 'q_icosahedral']:
                var_data = np.clip(var_data, 0.0, None)

            output_var_name = f"{var}_icosahedral" if not var.endswith("_icosahedral") else var
            orig_attrs = self.ds_src[var].attrs

            new_attrs = orig_attrs.copy()
            new_attrs.update({
                "long_name": f"{orig_attrs.get('long_name', var)} Interpolated to Icosahedral Mesh",
                "coordinates": "longitude latitude",
                "mesh": "icosahedral_mesh"
            })

            data_vars_out[output_var_name] = (["height", "node"], var_data, new_attrs)

        print("[STAGE 5] Formatting CF-1.8/UGRID compliant output dataset buffers...")

        # Safe extraction for optional static physical parameters
        for static_field, default_attrs in [
            ('land_sea_mask', {"long_name": "Land-Sea Mask", "units": "fraction"}),
            ('elevation', {"long_name": "Elevation", "units": "meters"})
        ]:
            if static_field in self.ds_mesh:
                data_vars_out[static_field] = (["node"], self.ds_mesh[static_field].values, self.ds_mesh[static_field].attrs)

        # Standard Mesh Geometry Output
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
            "height": self.ds_src.height.values,
            "node": self.ds_mesh.node.values,
            "face": self.ds_mesh.face.values,
            "three": np.arange(3)
        }
        for time_coord in ['time', 'step', 'valid_time']:
            if time_coord in self.ds_src.coords:
                coords_out[time_coord] = self.ds_src[time_coord].values

        ds_output = xr.Dataset(
            data_vars=data_vars_out,
            coords=coords_out,
            attrs={
                "title": "GFS Height Profiles Interpolated to Unstructured Global Icosahedral Grid Layout",
                "source_dataset": self.ds_src.attrs.get("title", "Vertical Height Reanalysis File"),
                "conventions": "CF-1.8 UGRID-1.0"
            }
        )

        print(f"Serializing output dataset to NetCDF: {self.output_path}")
        ds_output.to_netcdf(self.output_path, format="NETCDF4")

        self.ds_mesh.close()
        self.ds_src.close()
        ds_output.close()
        print("SUCCESS: Completed horizontal transformation pipeline completely.\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Level Lat-Lon to Unstructured Icosahedral Mesh Interpolator.")
    parser.add_argument("-i", "--input", required=True, help="Path to input height level NetCDF file")
    parser.add_argument("-m", "--mesh", required=True, help="Path to master icosahedral mesh file containing static geography")
    parser.add_argument("-o", "--output", required=True, help="Destination path for compiled output file")
    args = parser.parse_args()

    interpolator = LatLonToIcosahedralInterpolator(
        input_weather_path=args.input,
        master_mesh_path=args.mesh,
        output_path=args.output
    )
    interpolator.run_interpolation_pipeline()

if __name__ == "__main__":
    main()
