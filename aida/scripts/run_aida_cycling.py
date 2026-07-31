#!/usr/bin/env python3
"""
scripts/run_aida_cycling.py
---------------------------
AIDA Cycling Script for Inverting & Forecasting Non-Hydrostatic Log-State Fields.
Reads NetCDF background state, standardizes using checkpoint stats, evaluates GNN,
un-normalizes using saved dataset statistics, and exports clean analysis state.
"""

import os
import argparse
import numpy as np
import torch
import netCDF4 as nc

from train_aida_surrogate import IcosahedralGNNSurrogate, LOG_STATE_VARS

# Thermodynamic Constants
R_D = 287.058


def safe_log_transform(t_array, p_array):
    """Converts absolute T and P into bounded log-states (Fallback if log-vars missing)."""
    t_safe = np.maximum(t_array, 150.0)      # Kelvin lower floor
    p_safe = np.maximum(p_array, 1e-4)       # Pa lower floor

    rho_safe = p_safe / (R_D * t_safe)
    rho_safe = np.maximum(rho_safe, 1e-6)

    ln_t = np.log(t_safe)
    ln_p = np.log(p_safe)
    ln_rho = np.log(rho_safe)

    return ln_t, ln_p, ln_rho


def run_cycling_inference(
    background_file: str,
    output_file: str,
    gnn_ckpt: str,
    graph_path_m4: str
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[AIDA RUN] Execution Device: {device}")

    # ==========================================
    # 1. LOAD CHECKPOINT & STATISTICS
    # ==========================================
    if not os.path.exists(gnn_ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {gnn_ckpt}")

    print(f"[AIDA INIT] Loading GNN checkpoint: {gnn_ckpt}")
    checkpoint = torch.load(gnn_ckpt, map_location=device, weights_only=False)

    stats = checkpoint['stats']
    var_names = checkpoint.get('var_names', LOG_STATE_VARS)
    edge_index = torch.load(graph_path_m4, weights_only=False).to(device)

    # Reconstruct Model architecture
    model = IcosahedralGNNSurrogate(
        edge_index=edge_index,
        in_vars=len(var_names),
        levels=32
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ==========================================
    # 2. INGEST BACKGROUND NETCDF FILE
    # ==========================================
    print(f"[AIDA RUN] Processing Background File: {background_file}")
    with nc.Dataset(background_file, 'r') as ds:
        raw_vars = {}
        for var in var_names:
            if var in ds.variables:
                raw_vars[var] = np.asarray(ds.variables[var][:], dtype=np.float32)
            else:
                print(f"[WARNING] Variable {var} missing in input file. Attempting on-the-fly log derivation...")
                # Derive from basic T and P if log fields are not pre-computed
                t_raw = ds.variables['t_icosahedral'][:]
                p_raw = ds.variables['p_icosahedral'][:]
                ln_t, ln_p, ln_rho = safe_log_transform(t_raw, p_raw)
                raw_vars['ln_t_icosahedral'] = ln_t
                raw_vars['ln_p_icosahedral'] = ln_p
                raw_vars['ln_rho_icosahedral'] = ln_rho

    # Construct input array matching: [Vars, Levels, Nodes]
    state_list = []
    for var in var_names:
        arr = raw_vars[var]
        mean = stats[var]['mean']
        std = stats[var]['std'] if stats[var]['std'] > 1e-6 else 1.0

        # Sanitize NaNs/Infs directly in raw background input
        arr_clean = np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean)
        
        # Standardize: Z-Score
        norm_arr = (arr_clean - mean) / std
        state_list.append(norm_arr)

    # Shape: [1, 7, 32, 2562]
    x_input = np.stack(state_list, axis=0)[np.newaxis, ...]
    x_tensor = torch.from_numpy(x_input).float().to(device)

    # Check input buffer for lingering NaNs
    if torch.isnan(x_tensor).any():
        print("[FATAL] Input state tensor contains NaNs prior to forward pass! Replacing with 0.0.")
        x_tensor = torch.nan_to_num(x_tensor, nan=0.0)

    # ==========================================
    # 3. GNN FORWARD PASS & UN-NORMALIZATION
    # ==========================================
    print(f"[AIDA GNN] Evaluating surrogate forecast model with input shape {list(x_tensor.shape)}...")
    with torch.no_grad():
        y_pred_norm = model(x_tensor)  # Shape: [1, 7, 32, 2562]

    # Convert back to NumPy CPU
    y_pred_norm = y_pred_norm.squeeze(0).cpu().numpy()

    # Un-normalize back to physical log-space values
    unnorm_state = {}
    for idx, var in enumerate(var_names):
        mean = stats[var]['mean']
        std = stats[var]['std'] if stats[var]['std'] > 1e-6 else 1.0

        # Un-normalize: x = norm * std + mean
        phys_val = (y_pred_norm[idx] * std) + mean

        # Physical clamping on log-state variables to prevent exponential/overflow issues
        if var == 'ln_t_icosahedral':
            phys_val = np.clip(phys_val, 4.95, 6.0)     # ~140 K to 403 K
        elif var == 'ln_p_icosahedral':
            phys_val = np.clip(phys_val, -5.0, 13.0)   # ~0.006 Pa to 440 kPa
        elif var == 'ln_rho_icosahedral':
            phys_val = np.clip(phys_val, -15.0, 2.0)

        # Final sanity check against any lingering NaNs in output
        nan_count = np.isnan(phys_val).sum()
        if nan_count > 0:
            print(f"[WARNING] {var} output contains {nan_count} NaNs. Filling with mean: {mean:.4f}")
            phys_val = np.nan_to_num(phys_val, nan=mean)

        unnorm_state[var] = phys_val

    # ==========================================
    # 4. EXPORT TO OUTPUT NETCDF
    # ==========================================
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Copy background file structure and update log variables
    with nc.Dataset(background_file, 'r') as src, nc.Dataset(output_file, 'w') as dst:
        # Copy dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        # Copy global attributes
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        dst.aida_status = "ANALYSIS_CYCLE_COMPLETE"
        dst.aida_gnn_applied = "TRUE"

        # Write variables safely
        for var_name, var_obj in src.variables.items():
            # Determine fill value safely based on data type
            fill_val = None
            if hasattr(var_obj, '_FillValue'):
                fill_val = var_obj._FillValue
            elif np.issubdtype(var_obj.datatype, np.floating):
                fill_val = np.nan

            # Create variable using source datatype and valid fill_value
            out_var = dst.createVariable(
                var_name,
                var_obj.datatype,
                var_obj.dimensions,
                fill_value=fill_val
            )

            # Copy existing variable attributes (excluding _FillValue as it's set above)
            out_var.setncatts({
                k: var_obj.getncattr(k) for k in var_obj.ncattrs() if k != '_FillValue'
            })

            # Assign updated GNN prediction or copy original source data
            if var_name in unnorm_state:
                out_var[:] = unnorm_state[var_name]
            else:
                out_var[:] = src.variables[var_name][:]

    print(f"[AIDA SUCCESS] Exported analysis file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run AIDA Cycling Inference")
    parser.add_argument("-b", "--background", required=True, help="Input NetCDF background file")
    parser.add_argument("-o", "--output_file", required=True, help="Output Analysis NetCDF file")
    parser.add_argument("-c", "--gnn_ckpt", required=True, help="Path to trained GNN checkpoint (.pt)")
    parser.add_argument("-g4", "--graph_path_m4", required=True, help="Path to M4 mesh edge_index (.pt)")
    parser.add_argument("-g3", "--graph_path_m3", default=None, help="Path to M3 mesh edge_index (optional)")

    args = parser.parse_args()

    run_cycling_inference(
        background_file=args.background,
        output_file=args.output_file,
        gnn_ckpt=args.gnn_ckpt,
        graph_path_m4=args.graph_path_m4
    )


if __name__ == "__main__":
    main()
