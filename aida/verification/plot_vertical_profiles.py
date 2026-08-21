#!/usr/bin/env python3
"""
plot_vertical_profiles.py
-------------------------
Generates Monthly Verification Graphics:
1. Monthly mean vertical profile plots (RMSE, BIAS, ACC vs. Level Height).
2. 2D Time-Level Heatmaps (ACC, RMSE, BIAS over time for each variable) with 
   robust handling for missing top-level reference layers.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_units(var: str) -> str:
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


def plot_monthly_mean_profiles(mean_csv: str, output_dir: str):
    """Plots Monthly Mean Vertical Profiles (RMSE, BIAS, ACC)."""
    if not os.path.exists(mean_csv):
        print(f"[WARNING] Mean CSV missing: '{mean_csv}'. Skipping profile plots...")
        return

    df = pd.read_csv(mean_csv)
    variables = df['Variable'].unique()
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    for var in variables:
        df_var = df[df['Variable'] == var].dropna(subset=['RMSE', 'ACC']).sort_values(by='Level', ascending=True)

        if df_var.empty:
            continue

        levels = df_var['Level'].values
        rmse = df_var['RMSE'].values
        bias = df_var['BIAS'].values
        acc = df_var['ACC'].values

        fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
        fig.suptitle(f"Monthly Mean Analysis Vertical Profile — Variable: [{var.upper()}]", fontsize=14, fontweight='bold')

        # 1. RMSE Profile
        axes[0].plot(rmse, levels, marker='o', linewidth=2, color='tab:blue')
        axes[0].set_title("Monthly Mean RMSE", fontsize=11)
        axes[0].set_xlabel(f"RMSE ({get_units(var)})")
        axes[0].set_ylabel("Level Height (m)")
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. BIAS Profile
        axes[1].plot(bias, levels, marker='s', linewidth=2, color='tab:red')
        axes[1].axvline(0.0, color='black', linestyle=':', alpha=0.7)
        axes[1].set_title("Monthly Mean BIAS", fontsize=11)
        axes[1].set_xlabel(f"Bias ({get_units(var)})")
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # 3. ACC Profile
        axes[2].plot(acc, levels, marker='^', linewidth=2, color='tab:green')
        axes[2].axvline(1.0, color='black', linestyle=':', alpha=0.7)
        axes[2].set_title("Monthly Mean ACC", fontsize=11)
        axes[2].set_xlabel("ACC")
        axes[2].set_xlim(-0.05, 1.05)
        axes[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"monthly_profile_{var}.png")
        plt.savefig(out_path, dpi=300)
        plt.show()
        plt.close()
        print(f"  -> Saved Monthly Mean Profile: '{out_path}'")


def plot_monthly_heatmaps(master_csv: str, output_dir: str):
    """Plots 2D Time vs Level Heatmaps for ACC, RMSE, and BIAS across the month."""
    if not os.path.exists(master_csv):
        print(f"[WARNING] Master CSV missing: '{master_csv}'. Skipping heatmaps...")
        return

    df = pd.read_csv(master_csv)
    variables = df['Variable'].unique()
    os.makedirs(output_dir, exist_ok=True)

    for var in variables:
        df_var = df[df['Variable'] == var]

        # Pivot tables: Rows = Levels, Columns = Timestamps
        acc_pivot = df_var.pivot(index='Level', columns='Timestamp', values='ACC').sort_index(ascending=True)
        rmse_pivot = df_var.pivot(index='Level', columns='Timestamp', values='RMSE').sort_index(ascending=True)
        bias_pivot = df_var.pivot(index='Level', columns='Timestamp', values='BIAS').sort_index(ascending=True)

        # Handle missing reference levels (e.g. Level 31 NaNs in GFS truth starting Jan 20)
        acc_pivot = acc_pivot.ffill(axis=0).bfill(axis=0).fillna(0.0)
        rmse_pivot = rmse_pivot.ffill(axis=0).bfill(axis=0).fillna(0.0)
        bias_pivot = bias_pivot.ffill(axis=0).bfill(axis=0).fillna(0.0)

        levels = acc_pivot.index.values
        timestamps = acc_pivot.columns.values
        x_indices = np.arange(len(timestamps))

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(f"Monthly Verification Heatmaps — Variable: [{var.upper()}]", fontsize=16, fontweight='bold')

        # 1. ACC Heatmap
        c0 = axes[0].pcolormesh(x_indices, levels, acc_pivot.values, cmap='RdYlGn', vmin=0.0, vmax=1.0, shading='nearest')
        axes[0].set_title("Anomaly Correlation Coefficient (ACC)", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Level Height (m)")
        fig.colorbar(c0, ax=axes[0], label="ACC Score")

        # 2. RMSE Heatmap
        c1 = axes[1].pcolormesh(x_indices, levels, rmse_pivot.values, cmap='viridis', shading='nearest')
        axes[1].set_title(f"Root Mean Square Error (RMSE) [{get_units(var)}]", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Level Height (m)")
        fig.colorbar(c1, ax=axes[1], label=f"RMSE ({get_units(var)})")

        # 3. BIAS Heatmap (Diverging Colormap Centered at 0)
        max_bias = np.nanmax(np.abs(bias_pivot.values))
        if np.isnan(max_bias) or max_bias == 0:
            max_bias = 1.0

        c2 = axes[2].pcolormesh(x_indices, levels, bias_pivot.values, cmap='coolwarm', vmin=-max_bias, vmax=max_bias, shading='nearest')
        axes[2].set_title(f"Mean Bias (Analysis - Reference) [{get_units(var)}]", fontsize=12, fontweight='bold')
        axes[2].set_ylabel("Level Height (m)")
        axes[2].set_xlabel("Time (Analysis Cycles)")
        fig.colorbar(c2, ax=axes[2], label=f"Bias ({get_units(var)})")

        # Configure X-axis ticks
        step = max(1, len(timestamps) // 12)
        axes[2].set_xticks(x_indices[::step])
        axes[2].set_xticklabels(timestamps[::step], rotation=45, ha='right', fontsize=9)

        plt.tight_layout()
        out_heatmap = os.path.join(output_dir, f"monthly_heatmap_{var}.png")
        plt.savefig(out_heatmap, dpi=300)
        plt.show()
        plt.close()
        print(f"  -> Saved Monthly Heatmap: '{out_heatmap}'")


def main():
    parser = argparse.ArgumentParser(description="Plot Monthly Verification Metrics and Heatmaps")
    parser.add_argument("-m", "--master_csv", default="output/monthly_verification_levels.csv", help="Master CSV file")
    parser.add_argument("-s", "--mean_csv", default="output/monthly_mean_verification_levels.csv", help="Monthly mean CSV file")
    parser.add_argument("-o", "--outdir", default="output/plots/monthly", help="Output directory for plots")
    args = parser.parse_args()

    plot_monthly_mean_profiles(args.mean_csv, args.outdir)
    plot_monthly_heatmaps(args.master_csv, args.outdir)


if __name__ == "__main__":
    main()
