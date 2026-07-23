#!/usr/bin/env python
import argparse
import os
import numpy as np
import xarray as xr
import warnings

# Silence the xarray future warnings about concat defaults
warnings.filterwarnings("ignore", category=FutureWarning)

def interpolate_missing_frame(file_minus_6: str, file_plus_6: str, output_file: str):
    print(f"[LOAD] Opening bounding frames:\n -> t-6h: {os.path.basename(file_minus_6)}\n -> t+6h: {os.path.basename(file_plus_6)}")
    
    ds_prev = xr.open_dataset(file_minus_6)
    ds_next = xr.open_dataset(file_plus_6)
    
    # 1. Determine target timestamp (midpoint between the two files)
    time_prev = ds_prev.time.values[0] if ds_prev.time.ndim > 0 else ds_prev.time.values
    time_next = ds_next.time.values[0] if ds_next.time.ndim > 0 else ds_next.time.values
    
    dt_delta = (time_next - time_prev) / 2
    target_time = time_prev + dt_delta
    
    print(f"[DATETIME] Bounded Range:\n -> Previous: {str(time_prev)[:19]}\n -> Next:     {str(time_next)[:19]}")
    print(f" -> Interpolating Target (t+0h): {str(target_time)[:19]}")
    
    # 2. Combine datasets into a single array along a explicit time index for interpolation
    ds_prev_indexed = ds_prev.expand_dims(time=[time_prev])
    ds_next_indexed = ds_next.expand_dims(time=[time_next])
    
    combined_ds = xr.concat([ds_prev_indexed, ds_next_indexed], dim="time")
    
    # 3. Perform linear horizontal multi-field time interpolation
    print("[INTERP] Synthesizing mid-point variables via linear timeline mapping...")
    ds_interp = combined_ds.interp(time=target_time, method="linear")
    
    # Re-expand time back into a 1-length dimension array to match standard file structures
    ds_interp = ds_interp.expand_dims(time=[target_time])
    
    # 4. Apply physical corrections to specific humidity 'q_icosahedral' to avoid mathematical undershoot
    if 'q_icosahedral' in ds_interp.data_vars:
        print(" -> Enforcing non-negativity clipping baseline constraints on 'q_icosahedral'...")
        ds_interp['q_icosahedral'].values = np.clip(ds_interp['q_icosahedral'].values, 0.0, None)

    # 5. Restore metadata descriptions and forward tracking variables
    print("[PACKAGE] Rebuilding global file attributes and structural coordinate variables...")
    ds_interp.attrs = ds_prev.attrs.copy()
    ds_interp.attrs["title"] = "GFS Vertical Height Profiles Synthesized/Interpolated Frame for 3D Icosahedral Mesh"
    
    # Ensure static variables (like face_nodes) retain original integer datatypes
    for var in ds_interp.data_vars:
        if var in ds_prev.data_vars and ds_prev[var].dtype != ds_interp[var].dtype:
            ds_interp[var] = ds_interp[var].astype(ds_prev[var].dtype)

    print(f"[SAVE] Serializing final netCDF structure to destination: {output_file}")
    ds_interp.to_netcdf(output_path := os.path.abspath(output_file), format="NETCDF4")
    
    ds_prev.close()
    ds_next.close()
    ds_interp.close()
    print(f"SUCCESS: Synthesized timeline frame compiled completely at '{output_path}'.\n")

def main():
    parser = argparse.ArgumentParser(description="Interpolate a missing weather grid frame at t+0h from t-6h and t+6h files.")
    parser.add_argument("-p", "--prev", required=True, help="Path to t-6h NetCDF file")
    parser.add_argument("-n", "--next", required=True, help="Path to t+6h NetCDF file")
    parser.add_argument("-o", "--output", required=True, help="Path to write the interpolated output NetCDF file")
    args = parser.parse_args()

    interpolate_missing_frame(
        file_minus_6=args.prev,
        file_plus_6=args.next,
        output_file=args.output
    )

if __name__ == "__main__":
    main()
