#!/usr/bin/env python3
"""
convert_netcdf_to_logstate.py
------------------------------
Converts raw NetCDF background files to log-state space (ln_t, ln_rho, ln_p).
If density (rho) is not present in the NetCDF, it is computed from p and T.
"""

import os
import glob
import argparse
import numpy as np
import xarray as xr

R_D = 287.058  # Gas constant for dry air (J/(kg*K))

# Dynamic key lookup maps
T_NAMES = ['t_icosahedral', 't', 'temperature', 'TMP', 'T']
P_NAMES = ['p_icosahedral', 'p', 'pressure', 'PRES', 'P']
RHO_NAMES = ['rho_icosahedral', 'rho', 'density', 'DEN']

def find_var(ds, name_list):
    for name in name_list:
        if name in ds.data_vars:
            return name
    return None

def convert_file(input_nc: str, output_nc: str):
    print(f"[AIDA CONVERT] Processing: {input_nc}")
    ds = xr.open_dataset(input_nc)
    new_ds = xr.Dataset(coords=ds.coords, attrs=ds.attrs)

    # 1. Locate key thermodynamic variables
    t_var = find_var(ds, T_NAMES)
    p_var = find_var(ds, P_NAMES)
    rho_var = find_var(ds, RHO_NAMES)

    if not t_var or not p_var:
        raise ValueError(f"Could not find required Temperature/Pressure variables in {input_nc}")

    # Extract values
    T_val = np.clip(ds[t_var].values, 1e-5, None)
    p_val = np.clip(ds[p_var].values, 1e-5, None)

    # 2. Compute or load density
    if rho_var:
        rho_val = np.clip(ds[rho_var].values, 1e-8, None)
        print(f"  -> Found existing density variable: '{rho_var}'")
    elif 'ln_rho_icosahedral' in ds.data_vars:
        rho_val = np.exp(ds['ln_rho_icosahedral'].values)
    else:
        print(f"  -> 'rho' missing. Computing density from p and T (rho = p / R_d / T)...")
        rho_val = p_val / (R_D * T_val)

    # 3. Create log-state variables
    dims = ds[t_var].dims
    new_ds['ln_t_icosahedral']   = (dims, np.log(T_val))
    new_ds['ln_p_icosahedral']   = (dims, np.log(p_val))
    new_ds['ln_rho_icosahedral'] = (dims, np.log(rho_val))

    print(f"  -> Generated 'ln_t_icosahedral'   (Mean: {np.nanmean(new_ds['ln_t_icosahedral'].values):.4f})")
    print(f"  -> Generated 'ln_p_icosahedral'   (Mean: {np.nanmean(new_ds['ln_p_icosahedral'].values):.4f})")
    print(f"  -> Generated 'ln_rho_icosahedral' (Mean: {np.nanmean(new_ds['ln_rho_icosahedral'].values):.4f})")

    # 4. Preserve passthrough dynamic variables (u, v, w, q, etc.)
    for var in ds.data_vars:
        if var not in [t_var, p_var, rho_var] and not var.startswith('ln_'):
            new_ds[var] = ds[var]

    os.makedirs(os.path.dirname(output_nc), exist_ok=True)
    new_ds.to_netcdf(output_nc)
    print(f"[AIDA CONVERT] Saved: {output_nc}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.nc")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"[ERROR] No files matching '{args.pattern}' in {args.input_dir}")
        return

    for f in files:
        out_path = os.path.join(args.output_dir, os.path.basename(f))
        convert_file(f, out_path)

if __name__ == "__main__":
    main()
