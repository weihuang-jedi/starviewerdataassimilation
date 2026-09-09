#!/usr/bin/env python3
"""
scripts/plot_cycling_drift.py
------------------------------
Plots domain-average temperature/pressure drift and vertical profile evolution 
across AIDA operational cycling runs.
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

OUTPUT_DIR = "output/cycling_logstate_run1"
PLOT_SAVE_PATH = "output/cycling_logstate_run1/aida_cycling_drift.png"

def plot_drift(output_dir: str, save_path: str):
    nc_files = sorted(glob.glob(os.path.join(output_dir, "aida_analysis_cycle_*.nc")))
    
    if not nc_files:
        # Fallback search if files are saved without the prefix
        nc_files = sorted(glob.glob(os.path.join(output_dir, "*.nc")))

    if not nc_files:
        print(f"[ERROR] No output NetCDF files found in '{output_dir}'")
        return

    print(f"[AIDA PLOT] Found {len(nc_files)} cycle output files. Processing metrics...")

    cycles = []
    t_means, t_stds = [], []
    p_means, p_stds = [], []
    t_profiles, p_profiles = [], []

    for idx, nc_file in enumerate(nc_files):
        ds = xr.open_dataset(nc_file)
        
        # 1. Un-log variables into physical units
        if 'ln_t_icosahedral' in ds:
            T_phys = np.exp(ds['ln_t_icosahedral'].values)  # Kelvin
        elif 't_icosahedral' in ds:
            T_phys = ds['t_icosahedral'].values
        else:
            raise KeyError("Temperature variable not found in dataset.")

        if 'ln_p_icosahedral' in ds:
            p_phys = np.exp(ds['ln_p_icosahedral'].values) / 100.0  # Convert Pa to hPa
        elif 'p_icosahedral' in ds:
            p_phys = ds['p_icosahedral'].values / 100.0
        else:
            raise KeyError("Pressure variable not found in dataset.")

        cycles.append(idx)
        
        # Overall domain statistics
        t_means.append(np.nanmean(T_phys))
        t_stds.append(np.nanstd(T_phys))
        p_means.append(np.nanmean(p_phys))
        p_stds.append(np.nanstd(p_phys))

        # Vertical profile means (assuming shape: [batch/time, levels, nodes] or [levels, nodes])
        if T_phys.ndim == 3:
            t_profile = np.nanmean(T_phys, axis=(0, 2))
            p_profile = np.nanmean(p_phys, axis=(0, 2))
        elif T_phys.ndim == 2:
            t_profile = np.nanmean(T_phys, axis=1)
            p_profile = np.nanmean(p_phys, axis=1)
        else:
            t_profile = np.nanmean(T_phys)
            p_profile = np.nanmean(p_phys)

        t_profiles.append(t_profile)
        p_profiles.append(p_profile)

    # Convert lists to arrays
    cycles = np.array(cycles)
    t_means = np.array(t_means)
    t_stds = np.array(t_stds)
    p_means = np.array(p_means)
    p_stds = np.array(p_stds)

    # -------------------------------------------------------------
    # Plot Generation
    # -------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AIDA Operational Cycling Drift & Stability Diagnostics", fontsize=16, fontweight='bold')

    # Subplot 1: Temperature Mean Drift Across Cycles
    axs[0, 0].plot(cycles, t_means, 'o-', color='crimson', linewidth=2, markersize=8, label='Mean T (K)')
    axs[0, 0].fill_between(cycles, t_means - t_stds, t_means + t_stds, color='crimson', alpha=0.15, label='±1 Std Dev')
    axs[0, 0].set_title("Domain Mean Temperature Drift", fontweight='bold')
    axs[0, 0].set_xlabel("Cycle Index")
    axs[0, 0].set_ylabel("Temperature (K)")
    axs[0, 0].set_xticks(cycles)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()

    # Subplot 2: Pressure Mean Drift Across Cycles
    axs[0, 1].plot(cycles, p_means, 's-', color='navy', linewidth=2, markersize=8, label='Mean Pressure (hPa)')
    axs[0, 1].fill_between(cycles, p_means - p_stds, p_means + p_stds, color='navy', alpha=0.15, label='±1 Std Dev')
    axs[0, 1].set_title("Domain Mean Pressure Drift", fontweight='bold')
    axs[0, 1].set_xlabel("Cycle Index")
    axs[0, 1].set_ylabel("Pressure (hPa)")
    axs[0, 1].set_xticks(cycles)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend()

    # Subplot 3: Vertical Temperature Profile Evolution
    t_profiles = np.array(t_profiles)
    num_levels = t_profiles.shape[1] if t_profiles.ndim > 1 else 1
    level_axis = np.arange(num_levels)

    for c in cycles:
        if t_profiles.ndim > 1:
            axs[1, 0].plot(t_profiles[c], level_axis, label=f"Cycle {c}")
    axs[1, 0].set_title("Vertical Temperature Profile by Cycle", fontweight='bold')
    axs[1, 0].set_xlabel("Mean Temperature (K)")
    axs[1, 0].set_ylabel("Vertical Level Index")
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend()

    # Subplot 4: Vertical Pressure Profile Evolution
    p_profiles = np.array(p_profiles)
    for c in cycles:
        if p_profiles.ndim > 1:
            axs[1, 1].plot(p_profiles[c], level_axis, label=f"Cycle {c}")
    axs[1, 1].set_title("Vertical Pressure Profile by Cycle", fontweight='bold')
    axs[1, 1].set_xlabel("Mean Pressure (hPa)")
    axs[1, 1].set_ylabel("Vertical Level Index")
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[AIDA PLOT] Successfully saved diagnostic plot to: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_drift(OUTPUT_DIR, PLOT_SAVE_PATH)
