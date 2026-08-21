#!/usr/bin/env python3
"""
plot_vertical_profiles.py
-------------------------
Reads AIDA level-by-level verification metrics CSV (verification_levels.t06z.csv)
and plots vertical profiles of RMSE, BIAS, and Anomaly Correlation Coefficient (ACC)
across all 32 model levels for each state variable.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_vertical_metrics(csv_path: str, output_dir: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Verification CSV not found at: '{csv_path}'")

    print(f"[AIDA PLOT] Reading verification metrics from: '{csv_path}'")
    df = pd.read_csv(csv_path)

    # Variables to plot
    variables = df['Variable'].unique()
    os.makedirs(output_dir, exist_ok=True)

    # Set up Matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------------------
    # Plot 1: Combined Vertical Profiles (3 Subplots per Variable)
    # -------------------------------------------------------------------------
    for var in variables:
        df_var = df[df['Variable'] == var].sort_values(by='Level', ascending=True)

        levels = df_var['Level'].values
        rmse = df_var['RMSE'].values
        bias = df_var['BIAS'].values
        acc = df_var['ACC'].values

        fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
        fig.suptitle(f"AIDA Analysis Vertical Profiles — Variable: [{var.upper()}]", fontsize=14, fontweight='bold')

        # 1. RMSE Profile
        axes[0].plot(rmse, levels, marker='o', linewidth=2, color='tab:blue', label='RMSE')
        axes[0].set_title("Root Mean Square Error (RMSE)", fontsize=11)
        axes[0].set_xlabel(f"RMSE ({get_units(var)})")
        axes[0].set_ylabel("Level Height (m)")
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. BIAS Profile
        axes[1].plot(bias, levels, marker='s', linewidth=2, color='tab:red', label='BIAS')
        axes[1].axvline(0.0, color='black', linestyle=':', alpha=0.7)
        axes[1].set_title("Mean Bias (Analysis - Reference)", fontsize=11)
        axes[1].set_xlabel(f"Bias ({get_units(var)})")
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # 3. ACC Profile
        axes[2].plot(acc, levels, marker='^', linewidth=2, color='tab:green', label='ACC')
        axes[2].axvline(1.0, color='black', linestyle=':', alpha=0.7)
        axes[2].set_title("Anomaly Correlation Coefficient (ACC)", fontsize=11)
        axes[2].set_xlabel("ACC")
        axes[2].set_xlim(-0.05, 1.05)
        axes[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        out_single_path = os.path.join(output_dir, f"vertical_profile_{var}.png")
        plt.savefig(out_single_path, dpi=300)
        plt.show()
        plt.close()
        print(f"  -> Saved vertical profile for '{var}' to: '{out_single_path}'")

    # -------------------------------------------------------------------------
    # Plot 2: Summary Dashboard (All Variables ACC and Normalized RMSE)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig.suptitle("AIDA Summary Vertical Profiles Across All State Variables", fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(variables)))

    for idx, var in enumerate(variables):
        df_var = df[df['Variable'] == var].sort_values(by='Level', ascending=True)
        levels = df_var['Level'].values
        acc = df_var['ACC'].values
        rel_diff = df_var['RelDiff (%)'].values

        # Left Panel: ACC across all variables
        axes[0].plot(acc, levels, marker='o', label=var.upper(), color=colors[idx], linewidth=2)

        # Right Panel: Relative RMSE Diff (%)
        axes[1].plot(rel_diff, levels, marker='s', label=var.upper(), color=colors[idx], linewidth=2)

    axes[0].set_title("Anomaly Correlation Coefficient (ACC)", fontsize=12)
    axes[0].set_xlabel("ACC Score")
    axes[0].set_ylabel("Level Height (m)")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].axvline(1.0, color='gray', linestyle='--')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower left')

    axes[1].set_title("Relative Error / Normalized RMSE (%)", fontsize=12)
    axes[1].set_xlabel("Relative Diff (%)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_xscale('log')  # Log scale to accommodate small q vs large p relative diffs
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    summary_path = os.path.join(output_dir, "vertical_profiles_summary_dashboard.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"  -> Saved Summary Dashboard to: '{summary_path}'")


def get_units(var: str) -> str:
    """Helper to return variable physical units."""
    units_map = {
        't': 'K',
        'u': 'm/s',
        'v': 'm/s',
        'w': 'm/s',
        'q': 'kg/kg',
        'p': 'Pa',
        'rho': 'kg/m³'
    }
    return units_map.get(var.lower(), 'units')


def main():
    parser = argparse.ArgumentParser(description="Plot Vertical Metric Profiles from Verification CSV")
    parser.add_argument(
        "-i", "--csv",
        default="output/verification_levels.t06z.csv",
        help="Path to level-by-level verification CSV file"
    )
    parser.add_argument(
        "-o", "--outdir",
        default="output/plots/profiles",
        help="Output directory for generated plots"
    )
    args = parser.parse_args()

    plot_vertical_metrics(args.csv, args.outdir)


if __name__ == "__main__":
    main()
