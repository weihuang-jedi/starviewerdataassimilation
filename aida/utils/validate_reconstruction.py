#!/usr/bin/env python3
"""
validate_reconstruction.py
--------------------------
Verification suite with Cosine-Latitude Weighting for reconstructed AIDA analysis.
Computes spatial RMSE, MAE, BIAS, ACC, RelDiff (%), Min, and Max for both test and reference fields.
"""

import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr


def compute_weighted_metrics(pred: np.ndarray, ref: np.ndarray, lat_weights: np.ndarray):
    """
    Computes cosine-latitude weighted spatial statistics with accurate 3D normalization.
    pred, ref: arrays of shape [N], [lat, lon], or [level, lat, lon]
    lat_weights: 1D array of shape [lat]
    """
    p_min, p_max = float(np.nanmin(pred)), float(np.nanmax(pred))
    r_min, r_max = float(np.nanmin(ref)), float(np.nanmax(ref))

    diff = pred - ref

    # Build full 3D weight array matching pred/ref shape
    if pred.ndim == 3:  # [level, lat, lon]
        n_lev, n_lat, n_lon = pred.shape
        w = np.broadcast_to(lat_weights[np.newaxis, :, np.newaxis], (n_lev, n_lat, n_lon))
    elif pred.ndim == 2:  # [lat, lon]
        n_lat, n_lon = pred.shape
        w = np.broadcast_to(lat_weights[:, np.newaxis], (n_lat, n_lon))
    else:  # 1D icosahedral nodes
        w = lat_weights

    w_sum = np.sum(w)

    rmse = float(np.sqrt(np.sum(w * (diff ** 2)) / w_sum))
    mae = float(np.sum(w * np.abs(diff)) / w_sum)
    bias = float(np.sum(w * diff) / w_sum)

    p_mean = np.sum(w * pred) / w_sum
    r_mean = np.sum(w * ref) / w_sum

    p_ano = pred - p_mean
    r_ano = ref - r_mean

    var_p = np.sum(w * (p_ano ** 2))
    var_r = np.sum(w * (r_ano ** 2))

    # Guard against zero variance (constant predicted field)
    if var_p < 1e-12 or var_r < 1e-12:
        acc = 0.0
    else:
        cov = np.sum(w * p_ano * r_ano)
        acc = float(cov / np.sqrt(var_p * var_r))

    rel_diff = float((rmse / (np.abs(r_mean) + 1e-8)) * 100.0)

    return rmse, mae, bias, acc, rel_diff, p_min, p_max, r_min, r_max


def run_verification(test_file: str, ref_file: str, output_csv: str):
    print("=" * 60)
    print("[AIDA VALIDATION] Running Verification Suite (Cos-Lat Weighted)")
    print(f"Test File : {test_file}")
    print(f"Ref File  : {ref_file}")
    print("=" * 60)

    ds_test = xr.open_dataset(test_file)
    ds_ref = xr.open_dataset(ref_file)

    lat_key = 'lat' if 'lat' in ds_test.coords else ('latitude' if 'latitude' in ds_test.coords else None)
    if lat_key and ds_test[lat_key].ndim == 1:
        lats = ds_test[lat_key].values
        weights = np.cos(np.radians(lats))
    else:
        num_spatial = ds_test.sizes.get('node', ds_test.sizes.get('lon', 1))
        weights = np.ones(num_spatial)

    vars_to_eval = ['t', 'u', 'v', 'w', 'q', 'p']
    summary_rows = []
    level_rows = []

    for v in vars_to_eval:
        test_var = v if v in ds_test else f"{v}_icosahedral"
        ref_var = v if v in ds_ref else f"{v}_icosahedral"

        if test_var not in ds_test or ref_var not in ds_ref:
            continue

        print(f"Evaluating: '{test_var}' vs Reference: '{ref_var}'...")
        arr_test = ds_test[test_var].values
        arr_ref = ds_ref[ref_var].values

        # Fix Log-State Temperature if stored as raw ln_t (< 10.0 K)
        if v == 't' and np.nanmax(arr_test) < 10.0:
            print("  -> Unpacking log-temperature [exp(ln_t)]...")
            arr_test = np.exp(arr_test)

        # 1. Overall Spatial Summary
        rmse, mae, bias, acc, rel_diff, p_min, p_max, r_min, r_max = compute_weighted_metrics(arr_test, arr_ref, weights)
        summary_rows.append({
            'Variable': v,
            'RMSE': rmse,
            'MAE': mae,
            'BIAS': bias,
            'ACC': acc,
            'RelDiff (%)': rel_diff,
            'Pred_Min': p_min,
            'Pred_Max': p_max,
            'Ref_Min': r_min,
            'Ref_Max': r_max
        })

        # 2. Level-by-Level Breakdown
        if arr_test.ndim >= 2:
            levels = ds_test.coords.get('height', ds_test.coords.get('level', range(arr_test.shape[0]))).values
            for l_idx, lvl in enumerate(levels):
                l_test = arr_test[l_idx]
                l_ref = arr_ref[l_idx]
                l_rmse, l_mae, l_bias, l_acc, l_reldiff, lp_min, lp_max, lr_min, lr_max = compute_weighted_metrics(l_test, l_ref, weights)
                level_rows.append({
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

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 110)
    print("OVERALL SPATIAL METRICS SUMMARY (COSINE-LATITUDE WEIGHTED)")
    print("=" * 110)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))
    print("=" * 110)

    if level_rows:
        df_levels = pd.DataFrame(level_rows)
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        df_levels.to_csv(output_csv, index=False)
        print(f"\nDetailed level-by-level metrics written to: '{output_csv}'")


def main():
    parser = argparse.ArgumentParser(description="Validate Reconstructed AIDA Analysis Fields")
    parser.add_argument("-i", "--input", required=True, help="Input reconstructed AIDA NetCDF file")
    parser.add_argument("-r", "--ref", required=True, help="Reference truth NetCDF file")
    parser.add_argument("-o", "--output", default="output/verification_levels.csv", help="Output level-by-level CSV file")
    args = parser.parse_args()

    run_verification(args.input, args.ref, args.output)


if __name__ == "__main__":
    main()
