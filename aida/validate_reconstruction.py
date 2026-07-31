#!/usr/bin/env python3
"""
Validation script for AIDA reconstructed analyses against GFS reference data.
Applies cosine-latitude weighting to eliminate polar cell area distortion.
"""

import argparse
import sys
import numpy as np
import xarray as xr
import pandas as pd


def compute_latitude_weights(lats):
    """
    Computes normalized cosine-latitude weights.
    
    Args:
        lats (np.ndarray): 1D array of latitude values in degrees.
        
    Returns:
        np.ndarray: 1D array of normalized weights summing to 1.
    """
    rad_lats = np.radians(lats)
    cos_lats = np.cos(rad_lats)
    weights = cos_lats / np.sum(cos_lats)
    return weights


def compute_weighted_metrics(pred, ref, lat_weights):
    """
    Computes latitude-weighted RMSE, MAE, BIAS, ACC, and Relative Difference.
    
    Args:
        pred (np.ndarray): Predicted values, shape (lat, lon) or (height, lat, lon)
        ref (np.ndarray): Reference values, shape matching pred
        lat_weights (np.ndarray): 1D array of latitude weights matching lat dimension
        
    Returns:
        tuple: (rmse, mae, bias, acc, rel_diff)
    """
    # Ensure mask for non-NaN values
    valid_mask = np.isfinite(pred) & np.isfinite(ref)
    if not np.any(valid_mask):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Broadcast latitude weights across longitude axis
    # Assumes shape is (..., lat, lon)
    if pred.ndim == 2:
        weights_2d = lat_weights[:, None] * np.ones((1, pred.shape[1]))
    elif pred.ndim == 3:
        weights_2d = lat_weights[None, :, None] * np.ones((pred.shape[0], 1, pred.shape[2]))
    else:
        weights_2d = lat_weights

    # Mask invalid points in weights
    weights_masked = np.where(valid_mask, weights_2d, 0.0)
    weight_sum = np.sum(weights_masked, axis=(-2, -1), keepdims=True)
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum) # Avoid division by zero
    norm_weights = weights_masked / weight_sum

    # 1. Latitude-Weighted BIAS
    diff = pred - ref
    bias = np.sum(diff * norm_weights, axis=(-2, -1))

    # 2. Latitude-Weighted MAE
    mae = np.sum(np.abs(diff) * norm_weights, axis=(-2, -1))

    # 3. Latitude-Weighted RMSE
    rmse = np.sqrt(np.sum((diff ** 2) * norm_weights, axis=(-2, -1)))

    # 4. Latitude-Weighted Anomaly Correlation Coefficient (ACC)
    # Compute weighted anomalies relative to weighted spatial means
    pred_mean = np.sum(pred * norm_weights, axis=(-2, -1), keepdims=True)
    ref_mean = np.sum(ref * norm_weights, axis=(-2, -1), keepdims=True)

    pred_anom = pred - pred_mean
    ref_anom = ref - ref_mean

    cov = np.sum(pred_anom * ref_anom * norm_weights, axis=(-2, -1))
    var_pred = np.sum((pred_anom ** 2) * norm_weights, axis=(-2, -1))
    var_ref = np.sum((ref_anom ** 2) * norm_weights, axis=(-2, -1))

    denom = np.sqrt(var_pred * var_ref)
    acc = np.where(denom > 1e-12, cov / denom, 0.0)

    # 5. Latitude-Weighted Relative Difference (%)
    ref_abs_mean = np.sum(np.abs(ref) * norm_weights, axis=(-2, -1))
    rel_diff = np.where(ref_abs_mean > 1e-12, (mae / ref_abs_mean) * 100.0, 0.0)

    return rmse, mae, bias, acc, rel_diff


def run_verification(input_file, ref_file, output_csv):
    print("=" * 56)
    print("[AIDA VALIDATION] Running Verification Suite (Cos-Lat Weighted)")
    print(f"Test File : {input_file}")
    print(f"Ref File  : {ref_file}")
    print("=" * 56)

    ds_test = xr.open_dataset(input_file)
    ds_ref = xr.open_dataset(ref_file)

    # Resolve latitude dimension and coordinates
    lat_key = 'lat' if 'lat' in ds_test.coords else 'latitude'
    lats = ds_test[lat_key].values
    lat_weights = compute_latitude_weights(lats)

    target_vars = ['t', 'u', 'v', 'w', 'q', 'p']
    overall_rows = []
    detailed_rows = []

    for var in target_vars:
        if var not in ds_test or var not in ds_ref:
            print(f"Skipping '{var}': Not present in both datasets.")
            continue

        print(f"Evaluating: '{var}' vs Reference: '{var}'...")

        pred_data = ds_test[var].values
        ref_data = ds_ref[var].values

        # Ensure height axis is first dimension if 3D
        if pred_data.ndim == 3:
            # Overall metrics across 3D atmospheric volume
            # Collapse height dimension with mean for overall summary
            rmse_all, mae_all, bias_all, acc_all, reldiff_all = compute_weighted_metrics(
                pred_data, ref_data, lat_weights
            )
            
            # Take mean across all vertical levels for overall summary
            overall_rows.append({
                'Variable': var,
                'RMSE': np.nanmean(rmse_all),
                'MAE': np.nanmean(mae_all),
                'BIAS': np.nanmean(bias_all),
                'ACC': np.nanmean(acc_all),
                'RelDiff (%)': np.nanmean(reldiff_all)
            })

            # Level-by-level evaluation
            heights = ds_test['height'].values if 'height' in ds_test else np.arange(pred_data.shape[0])
            for idx, h in enumerate(heights):
                p_level = pred_data[idx]
                r_level = ref_data[idx]
                
                rmse, mae, bias, acc, _ = compute_weighted_metrics(p_level, r_level, lat_weights)
                detailed_rows.append({
                    'Variable': var,
                    'Level': float(h),
                    'RMSE': float(rmse),
                    'MAE': float(mae),
                    'BIAS': float(bias),
                    'ACC': float(acc)
                })

        elif pred_data.ndim == 2:
            rmse, mae, bias, acc, reldiff = compute_weighted_metrics(pred_data, ref_data, lat_weights)
            overall_rows.append({
                'Variable': var,
                'RMSE': float(rmse),
                'MAE': float(mae),
                'BIAS': float(bias),
                'ACC': float(acc),
                'RelDiff (%)': float(reldiff)
            })

    # Display Overall Summary Table
    df_overall = pd.DataFrame(overall_rows)
    print("\n" + "=" * 70)
    print("OVERALL SPATIAL METRICS SUMMARY (COSINE-LATITUDE WEIGHTED)")
    print("=" * 70)
    print(df_overall.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))
    print("=" * 70)

    # Save level-by-level CSV report
    if detailed_rows:
        df_detailed = pd.DataFrame(detailed_rows)
        df_detailed.to_csv(output_csv, index=False)
        print(f"\nDetailed level-by-level metrics written to: '{output_csv}'")


def main():
    parser = argparse.ArgumentParser(description="AIDA Cosine-Latitude Weighted Verification")
    parser.add_argument("-i", "--input", required=True, help="Input reconstructed analysis NetCDF file")
    parser.add_argument("-r", "--reference", required=True, help="Reference truth GFS NetCDF file")
    parser.add_argument("-o", "--output", default="output/verification_levels.csv", help="Output level CSV file")
    
    args = parser.parse_args()
    run_verification(args.input, args.reference, args.output)


if __name__ == "__main__":
    main()
