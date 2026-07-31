#!/usr/bin/env python3
"""
convert_zarr_logstate.py
------------------------
Transforms native icosahedral Zarr stores from raw physical state variables
[t, u, v, w, q, p] to scale-invariant height-coordinate log-state variables:
[ln_T, u, v, w, q, ln_rho, ln_p].

Ideal Gas Law in Log-Space:
  Tv = T * (1 + 0.608 * q)
  ln_rho = ln(p) - ln(R_d) - ln(Tv)
"""

import os
import argparse
import zarr
import numpy as np

# Dry air gas constant J/(kg K)
R_DRY = 287.058

def convert_zarr_store(input_zarr, output_zarr, chunk_timesteps=10):
    if not os.path.exists(input_zarr):
        raise FileNotFoundError(f"Input Zarr store not found at: {input_zarr}")

    print(f"[AIDA CONVERTER] Opening source store: {input_zarr}")
    src_root = zarr.open(input_zarr, mode='r')
    
    # Verify required keys exist
    required = ['t_icosahedral', 'u_icosahedral', 'v_icosahedral', 'w_icosahedral', 'q_icosahedral', 'p_icosahedral']
    for req in required:
        if req not in src_root:
            raise KeyError(f"Required key '{req}' missing from input store!")

    print(f"[AIDA CONVERTER] Creating log-state target store: {output_zarr}")
    dst_root = zarr.open(output_zarr, mode='w')

    # Read base shape metadata
    sample_arr = src_root['t_icosahedral']
    shape = sample_arr.shape
    chunks = (min(chunk_timesteps, shape[0]),) + shape[1:]
    
    print(f"[AIDA CONVERTER] Dataset Shape: {shape} | Chunking: {chunks}")

    # Target log-variable names
    target_vars = [
        'ln_t_icosahedral',
        'u_icosahedral',
        'v_icosahedral',
        'w_icosahedral',
        'q_icosahedral',
        'ln_rho_icosahedral',
        'ln_p_icosahedral'
    ]

    # Initialize zarr arrays in output store
    for var in target_vars:
        dst_root.create_dataset(
            var,
            shape=shape,
            chunks=chunks,
            dtype='float32',
            overwrite=True
        )

    num_timesteps = shape[0]
    print(f"[AIDA CONVERTER] Starting conversion across {num_timesteps} timesteps...")

    for t_start in range(0, num_timesteps, chunk_timesteps):
        t_end = min(t_start + chunk_timesteps, num_timesteps)

        # 1. Read batch slice
        t_raw = np.asarray(src_root['t_icosahedral'][t_start:t_end], dtype=np.float32)
        u_raw = np.asarray(src_root['u_icosahedral'][t_start:t_end], dtype=np.float32)
        v_raw = np.asarray(src_root['v_icosahedral'][t_start:t_end], dtype=np.float32)
        w_raw = np.asarray(src_root['w_icosahedral'][t_start:t_end], dtype=np.float32)
        q_raw = np.asarray(src_root['q_icosahedral'][t_start:t_end], dtype=np.float32)
        p_raw = np.asarray(src_root['p_icosahedral'][t_start:t_end], dtype=np.float32)

        # 2. Clean non-finite or fill values safely
        t_clean = np.nan_to_num(t_raw, nan=263.0, posinf=263.0, neginf=263.0)
        p_clean = np.nan_to_num(p_raw, nan=50000.0, posinf=50000.0, neginf=50000.0)
        q_clean = np.nan_to_num(q_raw, nan=1e-6, posinf=1e-6, neginf=1e-6)
        
        # Enforce physical floor bounds prior to log-transforms
        t_clean = np.maximum(t_clean, 100.0)      # T >= 100 K
        p_clean = np.maximum(p_clean, 1.0)        # P >= 1.0 Pa
        q_clean = np.maximum(q_clean, 1e-8)       # q >= 1e-8 kg/kg

        # 3. Calculate Virtual Temperature (Tv) & Log Transforms
        tv = t_clean * (1.0 + 0.608 * q_clean)
        
        ln_t   = np.log(t_clean)
        ln_p   = np.log(p_clean)
        
        # ln(rho) = ln(p) - ln(R_d) - ln(Tv)
        ln_rho = ln_p - np.log(R_DRY) - np.log(tv)

        # 4. Write transformed chunks back to target Zarr store
        dst_root['ln_t_icosahedral'][t_start:t_end]   = ln_t.astype(np.float32)
        dst_root['u_icosahedral'][t_start:t_end]      = u_raw.astype(np.float32)
        dst_root['v_icosahedral'][t_start:t_end]      = v_raw.astype(np.float32)
        dst_root['w_icosahedral'][t_start:t_end]      = w_raw.astype(np.float32)
        dst_root['q_icosahedral'][t_start:t_end]      = q_clean.astype(np.float32)
        dst_root['ln_rho_icosahedral'][t_start:t_end] = ln_rho.astype(np.float32)
        dst_root['ln_p_icosahedral'][t_start:t_end]   = ln_p.astype(np.float32)

        print(f"  -> Converted timesteps [{t_start:04d} - {t_end:04d} / {num_timesteps:04d}]")

    print(f"\n[COMPLETE] Log-state Zarr store successfully written to: {output_zarr}")

def main():
    parser = argparse.ArgumentParser(description="Convert Zarr store variables to log-state space.")
    parser.add_argument("-i", "--input", default="../data/icosahedral_2023.zarr", help="Input Zarr store path")
    parser.add_argument("-o", "--output", default="../data/icosahedral_2023_logstate.zarr", help="Output Zarr store path")
    args = parser.parse_args()

    convert_zarr_store(args.input, args.output)

if __name__ == "__main__":
    main()

