#!/usr/bin/env python3
"""
validate_reconstruction.py
--------------------------
Monthly Batch Verification Suite with Cosine-Latitude Weighting for reconstructed AIDA analysis.
Processes all reconstructed analysis files across the month, generating both an aggregated master CSV
and a monthly mean verification summary.
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import xarray as xr


def compute_weighted_metrics(pred: np.ndarray, ref: np.ndarray, lat_weights: np.ndarray):
    """
    Computes cosine-latitude weighted spatial statistics while safely filtering NaN values.
    """
    # 1. Build mask of valid (non-NaN) points across both arrays
    valid_mask = ~np.isnan(pred) & ~np.isnan(ref)
    if not np.any(valid_mask):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    p_valid = pred[valid_mask]
    r_valid = ref[valid_mask]

    p_min, p_max = float(np.min(p_valid)), float(np.max(p_valid))
    r_min, r_max = float(np.min(r_valid)), float(np.max(r_valid))

    diff = p_valid - r_valid

    # 2. Build full 3D weight array matching shape, then filter with valid_mask
    if pred.ndim == 3:  # [level, lat, lon]
        n_lev, n_lat, n_lon = pred.shape
        w_full = np.broadcast_to(lat_weights[np.newaxis, :, np.newaxis], (n_lev, n_lat, n_lon))
    elif pred.ndim == 2:  # [lat, lon]
        n_lat, n_lon = pred.shape
        w_full = np.broadcast_to(lat_weights[:, np.newaxis], (n_lat, n_lon))
    else:  # 1D nodes
        w_full = lat_weights

    w = w_full[valid_mask]
    w_sum = np.sum(w)

    if w_sum < 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0, p_min, p_max, r_min, r_max

    rmse = float(np.sqrt(np.sum(w * (diff ** 2)) / w_sum))
    mae = float(np.sum(w * np.abs(diff)) / w_sum)
    bias = float(np.sum(w * diff) / w_sum)

    p_mean = np.sum(w * p_valid) / w_sum
    r_mean = np.sum(w * r_valid) / w_sum

    p_ano = p_valid - p_mean
    r_ano = r_valid - r_mean

    var_p = np.sum(w * (p_ano ** 2))
    var_r = np.sum(w * (r_ano ** 2))

    if var_p < 1e-12 or var_r < 1e-12:
        acc = 0.0
    else:
        cov = np.sum(w * p_ano * r_ano)
        acc = float(cov / np.sqrt(var_p * var_r))

    rel_diff = float((rmse / (np.abs(r_mean) + 1e-8)) * 100.0)

    return rmse, mae, bias, acc, rel_diff, p_min, p_max, r_min, r_max


def process_monthly_batch(input_dir: str, ref_dir: str, master_csv: str, monthly_mean_csv: str):
    # Find all reconstructed analysis files matching reconstructed_aida_analysis_YYYYMMDD.tHHz.1p00.nc
    pattern = os.path.join(input_dir, "reconstructed_aida_analysis_*.1p00.nc")
    test_files = sorted(glob.glob(pattern))

    if not test_files:
        raise FileNotFoundError(f"[ERROR] No reconstructed analysis files found matching pattern: {pattern}")

    print(f"[AIDA BATCH] Found {len(test_files)} analysis files to process.")

    vars_to_eval = ['t', 'u', 'v', 'w', 'q', 'p']
    all_level_rows = []

    for test_file in test_files:
        # Extract timestamp string (e.g., 20250106.t06z)
        filename = os.path.basename(test_file)
        match = re.search(r"(\d{8}\.t\d{2}z)", filename)
        if not match:
            continue
        timestamp_str = match.group(1)
        ymd, cycle = timestamp_str.split('.t')
        cycle_hr = cycle.replace('z', '')

        # Construct corresponding GFS truth reference path
        ref_file = os.path.join(ref_dir, f"gfs.{ymd}.t{cycle_hr}z.1p00.f000.nc")
        if not os.path.exists(ref_file):
            print(f"[AIDA WARNING] Reference file missing for {timestamp_str}: '{ref_file}'. Skipping...")
            continue

        print(f"[PROCESSING] {timestamp_str} | Test: '{filename}' <-> Ref: 'gfs.{ymd}.t{cycle_hr}z.1p00.f000.nc'")

        ds_test = xr.open_dataset(test_file)
        ds_ref = xr.open_dataset(ref_file)

        lat_key = 'lat' if 'lat' in ds_test.coords else ('latitude' if 'latitude' in ds_test.coords else None)
        if lat_key and ds_test[lat_key].ndim == 1:
            lats = ds_test[lat_key].values
            weights = np.cos(np.radians(lats))
        else:
            num_spatial = ds_test.sizes.get('node', ds_test.sizes.get('lon', 1))
            weights = np.ones(num_spatial)

        for v in vars_to_eval:
            test_var = v if v in ds_test else f"{v}_icosahedral"
            ref_var = v if v in ds_ref else f"{v}_icosahedral"

            if test_var not in ds_test or ref_var not in ds_ref:
                continue

            arr_test = ds_test[test_var].values
            arr_ref = ds_ref[ref_var].values

            if v == 't' and np.nanmax(arr_test) < 10.0:
                arr_test = np.exp(arr_test)

            if arr_test.ndim >= 2:
                levels = ds_test.coords.get('height', ds_test.coords.get('level', range(arr_test.shape[0]))).values
                for l_idx, lvl in enumerate(levels):
                    l_test = arr_test[l_idx]
                    l_ref = arr_ref[l_idx]
                    l_rmse, l_mae, l_bias, l_acc, l_reldiff, lp_min, lp_max, lr_min, lr_max = compute_weighted_metrics(l_test, l_ref, weights)
                    
                    all_level_rows.append({
                        'Timestamp': timestamp_str,
                        'Date': ymd,
                        'Cycle': cycle_hr,
                        'Variable': v,
                        'Level': lvl,
                        'RMSE': l_rmse,
                        'MAE': l_mae,
                        'BIAS': l_bias,
                        'ACC': l_acc,
                        'RelDiff (%)': l_reldiff,
                        'Pred_Min': lp_min,
                        'Pred_Max': lp_max,
                        'Ref_Min': lr_min,
                        'Ref_Max': lr_max
                    })

        ds_test.close()
        ds_ref.close()

    # Save aggregated master CSV
    df_all = pd.DataFrame(all_level_rows)
    os.makedirs(os.path.dirname(master_csv) or '.', exist_ok=True)
    df_all.to_csv(master_csv, index=False)
    print(f"\n[AIDA SUCCESS] Master monthly metrics written to: '{master_csv}'")

    # Calculate and save monthly mean profiles
    df_monthly_mean = df_all.groupby(['Variable', 'Level'], as_index=False)[['RMSE', 'MAE', 'BIAS', 'ACC', 'RelDiff (%)']].mean()
    os.makedirs(os.path.dirname(monthly_mean_csv) or '.', exist_ok=True)
    df_monthly_mean.to_csv(monthly_mean_csv, index=False)
    print(f"[AIDA SUCCESS] Monthly mean vertical profiles written to: '{monthly_mean_csv}'")


def main():
    parser = argparse.ArgumentParser(description="Monthly Batch Verification Suite for Reconstructed AIDA Analysis")
    parser.add_argument("-i", "--indir", default="output", help="Directory containing reconstructed analysis files")
    parser.add_argument("-r", "--refdir", default="../data/regular_truth", help="Directory containing GFS truth files")
    parser.add_argument("-m", "--master_csv", default="output/monthly_verification_levels.csv", help="Master output CSV file")
    parser.add_argument("-s", "--mean_csv", default="output/monthly_mean_verification_levels.csv", help="Monthly mean output CSV file")
    args = parser.parse_args()

    process_monthly_batch(args.indir, args.refdir, args.master_csv, args.mean_csv)


if __name__ == "__main__":
    main()
