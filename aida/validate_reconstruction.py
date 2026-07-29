#!/usr/bin/env python3
import os
import argparse
import json
import xarray as xr
import pandas as pd
from verification_metrics import evaluate_variable

# Variable mapping in case short names differ between test and ref datasets
VAR_MAP = {
    "t_analysis": "t",
    "u_analysis": "u",
    "v_analysis": "v",
    "q_analysis": "q",
    "p_analysis": "p",
    "t_background": "t",
    "u_background": "u",
    "v_background": "v",
}

def main():
    parser = argparse.ArgumentParser(description="Validate regridded/reconstructed NetCDF data against reference grid.")
    parser.add_argument("-i", "--input", required=True, help="Input reconstructed regular grid NetCDF file")
    parser.add_argument("-r", "--ref", required=True, help="Reference truth NetCDF file (e.g., GFS reference)")
    parser.add_argument("-o", "--output", default="verification_summary.csv", help="CSV destination for level-by-level metrics")
    
    args = parser.parse_args()

    print(f"\n========================================================")
    print(f"[AIDA VALIDATION] Running Verification Suite")
    print(f"Test File : {args.input}")
    print(f"Ref File  : {args.ref}")
    print(f"========================================================\n")

    ds_test = xr.open_dataset(args.input)
    ds_ref = xr.open_dataset(args.ref)

    lats = ds_test['lat'].values if 'lat' in ds_test else None

    all_summaries = []
    level_rows = []

    # Iterate over variables present in the test dataset
    for test_var in ds_test.data_vars:
        # Determine matching variable name in reference dataset
        ref_var = VAR_MAP.get(test_var, test_var)

        if ref_var not in ds_ref:
            # Fallback check
            clean_name = test_var.replace("_analysis", "").replace("_background", "")
            if clean_name in ds_ref:
                ref_var = clean_name
            else:
                print(f"Skipping '{test_var}': Corresponding variable '{ref_var}' not found in reference dataset.")
                continue

        print(f"Evaluating: '{test_var}' vs Reference: '{ref_var}'...")

        test_da = ds_test[test_var]
        ref_da = ds_ref[ref_var]

        # Evaluate metrics
        res = evaluate_variable(test_da, ref_da, lats)
        overall = res["overall"]

        all_summaries.append({
            "Variable": test_var,
            "RMSE": overall["RMSE"],
            "MAE": overall["MAE"],
            "BIAS": overall["BIAS"],
            "ACC": overall["ACC"],
            "RelDiff (%)": overall["RelDiff_pct"]
        })

        # Process vertical breakdown if available
        for lvl_res in res["levels"]:
            level_rows.append({
                "Variable": test_var,
                "Level": lvl_res["level"],
                "RMSE": lvl_res["RMSE"],
                "MAE": lvl_res["MAE"],
                "BIAS": lvl_res["BIAS"],
                "ACC": lvl_res["ACC"],
            })

    # Display overall summary table in terminal
    df_summary = pd.DataFrame(all_summaries)
    print("\n" + "=" * 70)
    print("OVERALL SPATIAL METRICS SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)

    # Save detailed vertical level breakdown to CSV
    if level_rows:
        df_levels = pd.DataFrame(level_rows)
        df_levels.to_csv(args.output, index=False)
        print(f"\nDetailed level-by-level metrics written to: '{args.output}'")


if __name__ == "__main__":
    main()
