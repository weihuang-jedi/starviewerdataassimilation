#!/usr/bin/env python3
"""
validate_reconstruction.py
--------------------------
Verification and Validation Suite for AI-DA Cycling Outputs.
Evaluates regridded AI surrogate analysis/forecast states against reference truth datasets (e.g., GFS).
Handles unit harmonization (Pa <-> hPa, kg/kg <-> g/kg), layer-by-layer metrics, and empty/NaN slice masking.
"""

import os
import argparse
import numpy as np
import xarray as xr
import pandas as pd

# ==========================================
# VERIFICATION METRICS FUNCTIONS
# ==========================================
def compute_rmse(pred, ref):
    diff = pred - ref
    sq_err = diff ** 2
    if np.isnan(sq_err).all():
        return np.nan
    return float(np.sqrt(np.nanmean(sq_err)))

def compute_mae(pred, ref):
    abs_err = np.abs(pred - ref)
    if np.isnan(abs_err).all():
        return np.nan
    return float(np.nanmean(abs_err))

def compute_bias(pred, ref):
    diff = pred - ref
    if np.isnan(diff).all():
        return np.nan
    return float(np.nanmean(diff))

def compute_acc(pred, ref):
    """Anomaly Correlation Coefficient (ACC)."""
    mask = ~np.isnan(pred) & ~np.isnan(ref)
    if not np.any(mask):
        return np.nan
    
    p_clean = pred[mask]
    r_clean = ref[mask]
    
    p_prime = p_clean - np.mean(p_clean)
    r_prime = r_clean - np.mean(r_clean)
    
    denom = np.sqrt(np.sum(p_prime ** 2) * np.sum(r_prime ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(p_prime * r_prime) / denom)

def compute_rel_diff(rmse, ref):
    """Computes relative difference percentage against reference standard deviation or scale."""
    ref_scale = np.nanmean(np.abs(ref))
    if np.isnan(ref_scale) or ref_scale == 0:
        return np.nan
    return float((rmse / ref_scale) * 100.0)

# ==========================================
# UNIT HARMONIZATION & MAPPING HELPER
# ==========================================
def harmonize_units(pred_arr, ref_arr, var_name):
    """
    Standardizes units between predictions and reference truth files.
    - Pressure (p): Detects Pa vs. hPa scale mismatches.
    - Moisture (q): Detects kg/kg vs. g/kg scale mismatches.
    """
    pred = np.copy(pred_arr)
    ref = np.copy(ref_arr)
    
    # 1. Pressure Alignment (Pa -> hPa if reference is in hPa)
    if var_name == 'p':
        pred_mean = np.nanmean(pred)
        ref_mean = np.nanmean(ref)
        if pred_mean > 5000.0 and ref_mean < 2000.0:
            pred = pred / 100.0  # Pa to hPa

    # 2. Specific Humidity Alignment (kg/kg -> g/kg if reference is in g/kg)
    elif var_name == 'q':
        pred_mean = np.nanmean(pred)
        ref_mean = np.nanmean(ref)
        if pred_mean < 0.1 and ref_mean > 0.5:
            pred = pred * 1000.0  # kg/kg to g/kg
        elif pred_mean > 0.5 and ref_mean < 0.1:
            ref = ref * 1000.0    # align reference

    return pred, ref

# ==========================================
# MAIN EVALUATION PIPELINE
# ==========================================
def run_validation(input_file, ref_file, output_csv):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input verification file not found: {input_file}")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference truth file not found: {ref_file}")

    print("========================================================")
    print("[AIDA VALIDATION] Running Verification Suite")
    print(f"Test File : {input_file}")
    print(f"Ref File  : {ref_file}")
    print("========================================================\n")

    ds_test = xr.open_dataset(input_file)
    ds_ref  = xr.open_dataset(ref_file)

    target_vars = ['t', 'u', 'v', 'w', 'q', 'p']
    detailed_rows = []
    summary_rows = []

    for var in target_vars:
        if var not in ds_test or var not in ds_ref:
            print(f"Skipping '{var}': Not present in both test and reference datasets.")
            continue

        print(f"Evaluating: '{var}' vs Reference: '{var}'...")
        da_test = ds_test[var]
        da_ref  = ds_ref[var]

        # Squeeze out single time or ensemble dimensions
        raw_test = np.squeeze(da_test.values)
        raw_ref  = np.squeeze(da_ref.values)

        # Apply unit harmonization
        test_val, ref_val = harmonize_units(raw_test, raw_ref, var)

        # Global variable metrics across full 3D domain
        glob_rmse = compute_rmse(test_val, ref_val)
        glob_mae  = compute_mae(test_val, ref_val)
        glob_bias = compute_bias(test_val, ref_val)
        glob_acc  = compute_acc(test_val, ref_val)
        glob_rel  = compute_rel_diff(glob_rmse, ref_val)

        summary_rows.append({
            'Variable': var,
            'RMSE': glob_rmse,
            'MAE': glob_mae,
            'BIAS': glob_bias,
            'ACC': glob_acc,
            'RelDiff (%)': glob_rel
        })

        # Height/Level-by-Level Verification
        if 'height' in ds_test[var].dims or (test_val.ndim == 3):
            heights = ds_test['height'].values if 'height' in ds_test.coords else np.arange(test_val.shape[0])
            for h_idx, h_val in enumerate(heights):
                layer_test = test_val[h_idx]
                layer_ref  = ref_val[h_idx]

                # Check if layer is completely unpopulated/NaN
                if np.isnan(layer_ref).all() or np.isnan(layer_test).all():
                    l_rmse, l_mae, l_bias, l_acc = np.nan, np.nan, np.nan, np.nan
                else:
                    l_rmse = compute_rmse(layer_test, layer_ref)
                    l_mae  = compute_mae(layer_test, layer_ref)
                    l_bias = compute_bias(layer_test, layer_ref)
                    l_acc  = compute_acc(layer_test, layer_ref)

                detailed_rows.append({
                    'Variable': var,
                    'Level': float(h_val),
                    'RMSE': l_rmse,
                    'MAE': l_mae,
                    'BIAS': l_bias,
                    'ACC': l_acc
                })

    # Display Overall Summary Table
    df_summary = pd.DataFrame(summary_rows)
    print("\n======================================================================")
    print("OVERALL SPATIAL METRICS SUMMARY (HARMONIZED UNITS)")
    print("======================================================================")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))
    print("======================================================================\n")

    # Export Level-by-Level Metrics to CSV
    if output_csv:
        df_detailed = pd.DataFrame(detailed_rows)
        df_detailed.to_csv(output_csv, index=False)
        print(f"Detailed level-by-level metrics written to: '{output_csv}'")

    ds_test.close()
    ds_ref.close()

def main():
    parser = argparse.ArgumentParser(description="Validate regridded AI forecast against reference files.")
    parser.add_argument("-i", "--input", required=True, help="Input regridded NetCDF file")
    parser.add_argument("-r", "--ref", required=True, help="Reference NetCDF file (GFS standard)")
    parser.add_argument("-o", "--output", required=False, default="verification_levels.csv", help="Output CSV path")

    args = parser.parse_args()
    run_validation(args.input, args.ref, args.output)

if __name__ == "__main__":
    main()
