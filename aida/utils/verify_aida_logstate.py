#!/usr/bin/env python3
"""
verify_aida_logstate.py
-----------------------
Standalone verification harness for AI-DA Cycling outputs.
Computes RMSE, MAE, BIAS, and Anomaly Correlation Coefficient (ACC)
across vertical height levels, converting log-transformed outputs back
to physical units [T (K), u (m/s), v (m/s), w (m/s), q (kg/kg), p (Pa)].
"""

import os
import argparse
import numpy as np
import xarray as xr
import pandas as pd

HEIGHT_LEVELS = [
    2.0, 10.0, 20.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0,
    750.0, 1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0,
    4500.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11500.0,
    13000.0, 15000.0, 17500.0, 20000.0
]

def calculate_acc(pred, target):
    """Computes Anomaly Correlation Coefficient across spatial nodes."""
    pred_prime = pred - np.mean(pred)
    target_prime = target - np.mean(target)

    num = np.sum(pred_prime * target_prime)
    denom = np.sqrt(np.sum(pred_prime ** 2) * np.sum(target_prime ** 2))

    if denom < 1e-12:
        return 0.0
    return float(num / denom)


def evaluate_forecast_vs_truth(analysis_nc, truth_nc):
    if not os.path.exists(analysis_nc) or not os.path.exists(truth_nc):
        raise FileNotFoundError("Analysis or Truth NetCDF file missing for verification!")

    ds_an = xr.open_dataset(analysis_nc)
    ds_tr = xr.open_dataset(truth_nc)

    records = []

    # Target variable set
    var_pairs = [
        ('ln_t_icosahedral', 't'),
        ('u_icosahedral', 'u'),
        ('v_icosahedral', 'v'),
        ('w_icosahedral', 'w'),
        ('q_icosahedral', 'q'),
        ('ln_p_icosahedral', 'p')
    ]

    for log_var, short_var in var_pairs:
        if log_var not in ds_an.data_vars or log_var not in ds_tr.data_vars:
            continue

        an_data = ds_an[log_var].values
        tr_data = ds_tr[log_var].values

        # Convert log-variables back to physical space for verification
        if log_var == 'ln_t_icosahedral':
            an_phys = np.exp(an_data)
            tr_phys = np.exp(tr_data)
        elif log_var == 'ln_p_icosahedral':
            an_phys = np.exp(an_data)
            tr_phys = np.exp(tr_data)
        else:
            an_phys = an_data
            tr_phys = tr_data

        num_levels = min(an_phys.shape[0], len(HEIGHT_LEVELS))

        for lvl_idx in range(num_levels):
            lvl_height = HEIGHT_LEVELS[lvl_idx]
            p_slice = an_phys[lvl_idx]
            t_slice = tr_phys[lvl_idx]

            rmse = float(np.sqrt(np.mean((p_slice - t_slice) ** 2)))
            mae  = float(np.mean(np.abs(p_slice - t_slice)))
            bias = float(np.mean(p_slice - t_slice))
            acc  = calculate_acc(p_slice, t_slice)

            records.append({
                'Variable': short_var,
                'Level': lvl_height,
                'RMSE': rmse,
                'MAE': mae,
                'BIAS': bias,
                'ACC': acc
            })

    df = pd.DataFrame(records)

    print("\n======================================================================")
    print("LOG-STATE SPATIAL METRICS SUMMARY (PHYSICAL UNITS)")
    print("======================================================================")
    summary = df.groupby('Variable')[['RMSE', 'MAE', 'BIAS', 'ACC']].mean()
    print(summary.to_string())
    print("======================================================================\n")

    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate AIDA Log-State Cycling Performance.")
    parser.add_argument("-a", "--analysis", required=True, help="Path to analysis NetCDF file")
    parser.add_argument("-t", "--truth", required=True, help="Path to truth/verification NetCDF file")
    parser.add_argument("-o", "--output", default="logstate_verification_metrics.csv", help="Output CSV summary path")

    args = parser.parse_args()
    df_metrics = evaluate_forecast_vs_truth(args.analysis, args.truth)
    df_metrics.to_csv(args.output, index=False)
    print(f"[AIDA VERIFICATION] Full level metrics exported to: '{args.output}'")

if __name__ == "__main__":
    main()
