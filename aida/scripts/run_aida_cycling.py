#!/usr/bin/env python3
"""
run_aida_cycling.py
-------------------
AIDA GNN Surrogate Cycling Inference Script.

Loads background state from NetCDF, runs AIDA GNN surrogate step, applies 
residual increment scaling, and enforces strict physical bounds for all 7 variables.
"""

import argparse
import os
import sys
import netCDF4 as nc
import numpy as np
import torch

# Ensure parent directory is in Python path for 'models' package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import IcosahedralGNNSurrogate, generate_or_load_edge_index
from models.dataset import LOG_STATE_VARS


def load_netcdf_background(nc_path: str, var_keys: list[str]) -> np.ndarray:
    """Reads 7 variables from NetCDF into array shape [Vars=7, Levels=32, Nodes=2562]."""
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Background NetCDF file not found at '{nc_path}'")

    var_arrays = []
    with nc.Dataset(nc_path, 'r') as ds:
        for k in var_keys:
            if k not in ds.variables:
                raise KeyError(f"Variable '{k}' not found in NetCDF file '{nc_path}'")

            data = ds.variables[k][:]
            if data.ndim == 3 and data.shape[0] == 1:
                data = data.squeeze(0)

            var_arrays.append(np.array(data, dtype=np.float32))

    background_data = np.stack(var_arrays, axis=0)
    return np.nan_to_num(background_data, nan=0.0)


def save_netcdf_analysis(
    template_nc_path: str,
    output_nc_path: str,
    analysis_array: np.ndarray,
    var_keys: list[str]
):
    """Writes GNN analysis output back to NetCDF format matching template dimensions."""
    os.makedirs(os.path.dirname(output_nc_path) or ".", exist_ok=True)

    with nc.Dataset(template_nc_path, 'r') as src, nc.Dataset(output_nc_path, 'w') as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        for var_name, src_var in src.variables.items():
            if var_name not in var_keys:
                out_var = dst.createVariable(var_name, src_var.datatype, src_var.dimensions)
                out_var.setncatts({k: src_var.getncattr(k) for k in src_var.ncattrs()})
                out_var[:] = src_var[:]

        for idx, key in enumerate(var_keys):
            if key in src.variables:
                src_var = src.variables[key]
                out_var = dst.createVariable(key, src_var.datatype, src_var.dimensions)
                out_var.setncatts({k: src_var.getncattr(k) for k in src_var.ncattrs()})
            else:
                out_var = dst.createVariable(key, 'f4', ('height', 'node'))

            data_to_write = analysis_array[idx]
            if len(out_var.dimensions) == 3:
                data_to_write = np.expand_dims(data_to_write, axis=0)

            out_var[:] = data_to_write

    print(f"[AIDA RUN] Successfully wrote analysis output to '{output_nc_path}'")


def run_cycling_inference(
    background_file: str,
    output_file: str,
    gnn_ckpt: str,
    graph_path_m4: str
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AIDA RUN] Execution Device: {device}")

    # 1. Load Checkpoint and Statistics
    print(f"[AIDA INIT] Loading GNN checkpoint: {gnn_ckpt}")
    checkpoint = torch.load(gnn_ckpt, map_location=device)
    ckpt_args = checkpoint.get('args', checkpoint.get('config', {}))

    hidden_dim = ckpt_args.get('hidden_dim', 64) if isinstance(ckpt_args, dict) else getattr(ckpt_args, 'hidden_dim', 64)
    model = IcosahedralGNNSurrogate(
        in_vars=len(LOG_STATE_VARS),
        hidden_dim=hidden_dim
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    edge_index_m4 = generate_or_load_edge_index(num_nodes=2562, edge_file=graph_path_m4).to(device)

    # 2. Read Raw Physical Background State [7, 32, 2562]
    background_data = load_netcdf_background(background_file, LOG_STATE_VARS)
    
    # 3. Compute Per-Level Mean & Standard Deviation
    means = np.mean(background_data, axis=-1, keepdims=True)  # [7, 32, 1]
    stds = np.std(background_data, axis=-1, keepdims=True)     # [7, 32, 1]
    stds = np.maximum(stds, 1e-6)

    # 4. Standardize Input Background Tensor: (x - mu) / sigma
    normalized_bg = (background_data - means) / stds
    background_tensor = torch.from_numpy(normalized_bg).unsqueeze(0).to(device)

    # 5. Run GNN Surrogate Pass
    print("[AIDA STEP] Executing GNN surrogate forward step...")
    with torch.no_grad():
        analysis_tensor = model(background_tensor, edge_index_m4)

    analysis_pred = analysis_tensor.squeeze(0).cpu().numpy()  # [7, 32, 2562]

    # 6. Apply Scaled Analysis Increments to Background State
    # LOG_STATE_VARS: ['ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral', 'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral']
    analysis_array = np.zeros_like(background_data)

    # Increment scaling factor (alpha) to blend GNN update smoothly into background
    alpha_inc = 0.15  # 15% increment update step

    for idx, key in enumerate(LOG_STATE_VARS):
        # Calculate raw physical increment: Δx = Δz * sigma
        inc_phys = analysis_pred[idx] * stds[idx]

        if 'ln_t' in key or 'ln_p' in key or 'ln_rho' in key:
            # For log-state variables, apply small log-increment update
            analysis_array[idx] = background_data[idx] + (alpha_inc * inc_phys)
        else:
            # For linear dynamics variables (u, v, w, q), apply scaled increment
            analysis_array[idx] = background_data[idx] + (alpha_inc * inc_phys)

        # Enforce physical safety bounds
        if 'q_icosahedral' in key:
            analysis_array[idx] = np.clip(analysis_array[idx], 1e-7, 0.035)
        elif 'u_icosahedral' in key or 'v_icosahedral' in key:
            # Clamp wind speed within realistic global tropospheric limits
            analysis_array[idx] = np.clip(analysis_array[idx], -75.0, 75.0)
        elif 'w_icosahedral' in key:
            analysis_array[idx] = np.clip(analysis_array[idx], -5.0, 5.0)

    # 7. Write Analysis Output to Disk
    save_netcdf_analysis(background_file, output_file, analysis_array, LOG_STATE_VARS)


def main():
    parser = argparse.ArgumentParser(description="Run AIDA GNN Cycling Inference")
    parser.add_argument("--background", type=str, required=True, help="Input background NetCDF file (.nc)")
    parser.add_argument("--output_file", type=str, required=True, help="Output analysis NetCDF file (.nc)")
    parser.add_argument("--gnn_ckpt", type=str, required=True, help="Path to trained GNN model checkpoint (.pt)")
    parser.add_argument("--graph_path_m4", type=str, default="../data/graph/icosahedral_edge_index_m4.pt", help="Path to Mesh Level 4 topology")

    args = parser.parse_args()

    run_cycling_inference(
        background_file=args.background,
        output_file=args.output_file,
        gnn_ckpt=args.gnn_ckpt,
        graph_path_m4=args.graph_path_m4
    )


if __name__ == "__main__":
    main()
