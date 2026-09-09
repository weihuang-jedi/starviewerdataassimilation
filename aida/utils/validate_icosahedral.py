#!/usr/bin/env python3
"""
utils/validate_icosahedral.py
-----------------------------
Direct Native-Grid Verification Suite for AIDA AI-DA Analysis States.
Applies mu/std physical un-normalization transforms and calculates 
latitude-weighted RMSE, MAE, BIAS, ACC, and level-by-level vertical 
profiles directly on the 40,962-node unstructured icosahedral mesh.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

MU_LN_T = 5.50
STD_LN_T = 0.15

MU_LN_P = 10.50
STD_LN_P = 1.20

MU_Q = 0.005
STD_Q = 0.005


def compute_latitude_weights(lats_deg: np.ndarray) -> np.ndarray:
    """Computes normalized cosine latitude weights for unstructured nodes."""
    rad = np.pi / 180.0
    weights = np.cos(lats_deg * rad)
    return weights / np.sum(weights)


def unnormalize_variable(var_name: str, raw_arr: np.ndarray) -> np.ndarray:
    """Converts normalized model output tensors to true physical values."""
    mean_val = np.nanmean(raw_arr)

    if var_name in ['ln_t_icosahedral', 'ln_t', 't']:
        if abs(mean_val) < 3.0:
            ln_t_phys = raw_arr * STD_LN_T + MU_LN_T
            return np.exp(ln_t_phys)
        elif mean_val < 10.0:
            return np.exp(raw_arr)
        return raw_arr

    elif var_name in ['ln_p_icosahedral', 'ln_p', 'p']:
        if abs(mean_val) < 3.0:
            ln_p_phys = raw_arr * STD_LN_P + MU_LN_P
            return np.exp(ln_p_phys) / 100.0
        elif mean_val < 15.0:
            return np.exp(raw_arr) / 100.0
        elif mean_val > 10000.0:
            return raw_arr / 100.0
        return raw_arr

    elif var_name in ['q_icosahedral', 'q']:
        if abs(mean_val) < 3.0:
            return np.maximum(0.0, raw_arr * STD_Q + MU_Q)
        return raw_arr

    return raw_arr


def evaluate_native_icosahedral(pred_file: str, ref_file: str, output_csv: str = None):
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"[ERROR] Prediction file not found: '{pred_file}'")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"[ERROR] Reference truth file not found: '{ref_file}'")

    print("=" * 90, flush=True)
    print(f"[AIDA NATIVE VALIDATION] Evaluating Unstructured Icosahedral Grid Metrics")
    print(f" Analysis File : {pred_file}")
    print(f" Truth File    : {ref_file}")
    print("=" * 90, flush=True)

    ds_pred = xr.open_dataset(pred_file)
    ds_ref = xr.open_dataset(ref_file)

    lats = ds_ref['latitude'].values if 'latitude' in ds_ref else ds_pred['latitude'].values
    if lats.ndim > 1:
        lats = lats[0]

    weights = compute_latitude_weights(lats)  # [Nodes]

    var_map = [
        ('ln_t_icosahedral', 't', 'Temperature (K)'),
        ('u_icosahedral', 'u', 'U-Wind (m/s)'),
        ('v_icosahedral', 'v', 'V-Wind (m/s)'),
        ('w_icosahedral', 'w', 'W-Wind (Pa/s)'),
        ('q_icosahedral', 'q', 'Specific Humidity (kg/kg)'),
        ('ln_p_icosahedral', 'p', 'Pressure (hPa)'),
    ]

    level_csv_rows = []

    print(f"{'Variable':<12} | {'RMSE':>12} | {'MAE':>12} | {'BIAS':>12} | {'ACC':>10} | {'Pred Min':>10} | {'Pred Max':>10}")
    print("-" * 90, flush=True)

    for pred_var, ref_var, desc in var_map:
        p_name = pred_var if pred_var in ds_pred else (pred_var.replace('_icosahedral', '') if pred_var.replace('_icosahedral', '') in ds_pred else None)
        r_name = pred_var if pred_var in ds_ref else (pred_var.replace('_icosahedral', '') if pred_var.replace('_icosahedral', '') in ds_ref else None)

        if p_name is None or r_name is None:
            continue

        pred_val = ds_pred[p_name].values
        ref_val = ds_ref[r_name].values

        if pred_val.ndim == 3:
            pred_val = pred_val[0]
        if ref_val.ndim == 3:
            ref_val = ref_val[0]

        pred_phys = unnormalize_variable(p_name, pred_val)
        ref_phys = unnormalize_variable(r_name, ref_val)

        diff = pred_phys - ref_phys
        num_levels = pred_phys.shape[0]

        # 3D Metrics
        weights_3d = np.tile(weights[np.newaxis, :], (num_levels, 1))
        weights_3d = weights_3d / np.sum(weights_3d)

        bias = np.sum(diff * weights_3d)
        mae = np.sum(np.abs(diff) * weights_3d)
        rmse = np.sqrt(np.sum((diff ** 2) * weights_3d))

        pred_anomaly = pred_phys - np.mean(pred_phys)
        ref_anomaly = ref_phys - np.mean(ref_phys)
        cov = np.sum(pred_anomaly * ref_anomaly * weights_3d)
        var_p = np.sum((pred_anomaly ** 2) * weights_3d)
        var_r = np.sum((ref_anomaly ** 2) * weights_3d)
        acc = cov / (np.sqrt(var_p * var_r) + 1e-12)

        p_min, p_max = np.min(pred_phys), np.max(pred_phys)

        print(f"{ref_var:<12} | {rmse:12.4f} | {mae:12.4f} | {bias:12.4f} | {acc:10.4f} | {p_min:10.2f} | {p_max:10.2f}")

        # Level Breakdown
        for lev in range(num_levels):
            p_lev = pred_phys[lev, :]
            r_lev = ref_phys[lev, :]
            d_lev = p_lev - r_lev

            w_lev = weights / np.sum(weights)
            b_lev = np.sum(d_lev * w_lev)
            m_lev = np.sum(np.abs(d_lev) * w_lev)
            r_lev_metric = np.sqrt(np.sum((d_lev ** 2) * w_lev))

            p_anom = p_lev - np.mean(p_lev)
            r_anom = r_lev - np.mean(r_lev)
            cov_l = np.sum(p_anom * r_anom * w_lev)
            v_p_l = np.sum((p_anom ** 2) * w_lev)
            v_r_l = np.sum((r_anom ** 2) * w_lev)
            acc_l = cov_l / (np.sqrt(v_p_l * v_r_l) + 1e-12)

            rel_diff = (r_lev_metric / (np.mean(np.abs(r_lev)) + 1e-8)) * 100.0

            level_csv_rows.append({
                'Variable': ref_var,
                'Level': lev + 1,
                'RMSE': r_lev_metric,
                'MAE': m_lev,
                'BIAS': b_lev,
                'ACC': acc_l,
                'RelDiff (%)': rel_diff
            })

    print("=" * 90, flush=True)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_out = pd.DataFrame(level_csv_rows)
        df_out.to_csv(output_csv, index=False)
        print(f"[AIDA SUCCESS] Saved level verification metrics CSV to: '{output_csv}'", flush=True)

    ds_pred.close()
    ds_ref.close()


def main():
    parser = argparse.ArgumentParser(description="Direct Native Icosahedral Grid Verification")
    parser.add_argument("-i", "--input", required=True, help="Path to generated icosahedral NetCDF file")
    parser.add_argument("-r", "--reference", required=True, help="Path to reference truth icosahedral NetCDF file")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path for level-by-level metrics")
    args = parser.parse_args()

    evaluate_native_icosahedral(pred_file=args.input, ref_file=args.reference, output_csv=args.output)


if __name__ == "__main__":
    main()
