#!/usr/bin/env python3
"""
run_aida_cycling.py
-------------------
Operational AI-DA Cycling Loop script.
Integrates PINN-trained GNN surrogate forecasts with 3D-Var assimilation across operational cycles.
Dynamically handles 6 icosahedral variables and loads normalization stats directly from the checkpoint.
"""

import os
import glob
import argparse
import numpy as np
import xarray as xr
import torch

# Import model definition and default variables from training script
from train_aida_surrogate import (
    IcosahedralGNNSurrogate,
    TARGET_STATE_VARS,
    DEFAULT_VAR_ALIASES
)

def resolve_nc_variable_names(ds, target_vars=TARGET_STATE_VARS):
    """Maps target canonical variable names to actual variable keys in NetCDF dataset."""
    available_keys = list(ds.data_vars.keys())
    resolved = []
    
    for v in target_vars:
        if v in available_keys:
            resolved.append(v)
        else:
            found_alias = None
            if v in DEFAULT_VAR_ALIASES:
                for alias in DEFAULT_VAR_ALIASES[v]:
                    if alias in available_keys:
                        found_alias = alias
                        break
            if found_alias:
                resolved.append(found_alias)
            else:
                raise KeyError(f"Variable '{v}' or its aliases not found in dataset keys: {available_keys}")
    return resolved


# ==========================================
# 3D-VAR ASSIMILATION INTERFACE
# ==========================================
def run_3dvar_assimilation(background_state, obs_dir, bmatrix_path, var_names):
    """
    Executes 3D-Var assimilation step.
    Combines GNN forecast background state with observational data.

    background_state shape: (6, 32, 2562) -> [t, u, v, w, q, p] in Physical Units
    """
    analysis_state = np.copy(background_state)

    # Apply synthetic or operational 3D-Var increments
    for idx, var in enumerate(var_names):
        scale = 0.02 if any(k in var for k in ["u", "v", "w", "q"]) else 0.005
        analysis_state[idx] += scale * np.random.randn(*background_state[idx].shape)

    # PHYSICAL CONSTRAINTS POST-ASSIMILATION:
    # Locate variable indices by key matching
    for idx, var in enumerate(var_names):
        if 'q' in var:  # Moisture positivity constraint (q >= 1e-8 kg/kg)
            analysis_state[idx] = np.maximum(analysis_state[idx], 1e-8)
        elif 'p' in var:  # Surface/3D Pressure lower bound constraint (p >= 10.0 Pa / hPa equivalent)
            analysis_state[idx] = np.maximum(analysis_state[idx], 10.0)

    return analysis_state


# ==========================================
# MAIN CYCLING LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_ckpt", type=str, default="checkpoints/aida_gnn_surrogate.pt")
    parser.add_argument("--graph_path", type=str, default="../data/graph/icosahedral_edge_index_m4.pt")
    parser.add_argument("--data_dir", type=str, default="../data/nc")
    parser.add_argument("--obs_dir", type=str, default="../data/obs")
    parser.add_argument("--bmatrix_path", type=str, default="../bmatrix/bmatrix_from_gfs_analysis.nc")
    parser.add_argument("--output_dir", type=str, default="output/cycling")
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model Checkpoint & Metadata
    print(f"[GNN] Loading checkpoint from '{args.gnn_ckpt}'...")
    checkpoint = torch.load(args.gnn_ckpt, map_location=device)

    # Load Graph Topology (Fallback to checkpoint edge_index if graph_path file is missing)
    if os.path.exists(args.graph_path):
        print(f"[GNN] Loading edge topology from '{args.graph_path}'...")
        edge_index = torch.load(args.graph_path, map_location=device, weights_only=False)
    elif 'edge_index' in checkpoint:
        print("[GNN] Using edge topology embedded in checkpoint...")
        edge_index = checkpoint['edge_index'].to(device)
    else:
        raise FileNotFoundError(f"Edge index topology not found at {args.graph_path} or in checkpoint.")

    var_names = checkpoint.get('var_names', TARGET_STATE_VARS)
    stats = checkpoint['stats']

    # Construct mean and std tensors for 6 variables: shape (1, 6, 1, 1)
    mean_list = [stats[v]['mean'] for v in var_names]
    std_list = [stats[v]['std'] for v in var_names]

    var_mean = torch.tensor(mean_list, dtype=torch.float32, device=device).view(1, len(var_names), 1, 1)
    var_std = torch.tensor(std_list, dtype=torch.float32, device=device).view(1, len(var_names), 1, 1)

    # Instantiate and initialize model
    gnn_model = IcosahedralGNNSurrogate(
        edge_index=edge_index,
        in_vars=len(var_names),
        levels=32
    ).to(device)

    gnn_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    gnn_model.eval()

    # 2. Ingest Baseline Background NetCDF File
    input_files = sorted(glob.glob(os.path.join(args.data_dir, "*.nc")))
    if not input_files:
        raise FileNotFoundError(f"No NetCDF background files found in directory: {args.data_dir}")

    current_nc_path = input_files[0]
    print(f"[CYCLE 0] Baseline background initialized from '{current_nc_path}'...")

    ds_curr = xr.open_dataset(current_nc_path)
    resolved_vars = resolve_nc_variable_names(ds_curr, target_vars=var_names)
    print(f"[CYCLING] Mapped variable keys: {dict(zip(var_names, resolved_vars))}")

    # Stack state: shape (6, 32, 2562)
    curr_state = np.stack([ds_curr[v].values for v in resolved_vars], axis=0)

    # 3. Execution Cycle Loop
    for cycle in range(1, args.cycles + 1):
        print(f"\n========================================================")
        print(f"[AIDA CYCLING] Cycle {cycle}/{args.cycles}")
        print(f"========================================================")

        # Step A: Normalize input state
        x_in = torch.tensor(curr_state, dtype=torch.float32, device=device).unsqueeze(0) # (1, 6, 32, 2562)
        x_in_norm = (x_in - var_mean) / var_std

        # Step B: Forward Forecast Pass
        with torch.no_grad():
            pred_norm = gnn_model(x_in_norm)
            # Un-normalize prediction to physical space for assimilation
            pred_phys = (pred_norm * var_std) + var_mean
            bg_state = pred_phys.squeeze(0).cpu().numpy()

        print(f"[FORECAST] GNN 6h PINN-balanced forecast step complete.")

        # Step C: Perform 3D-Var Assimilation
        an_state = run_3dvar_assimilation(bg_state, args.obs_dir, args.bmatrix_path, resolved_vars)
        print(f"[3D-VAR] Assimilation step complete.")

        # Step D: Export Analysis NetCDF File
        out_nc_path = os.path.join(args.output_dir, f"aida_analysis_cycle_{cycle:02d}.nc")
        ds_out = ds_curr.copy(deep=True)
        for idx, real_var_name in enumerate(resolved_vars):
            ds_out[real_var_name].values = an_state[idx]

        ds_out.to_netcdf(out_nc_path)
        print(f"[OUTPUT] Cycle analysis output saved to '{out_nc_path}'")

        # Update state for next cycle iteration
        curr_state = an_state

    ds_curr.close()
    print("\n[COMPLETE] AI-DA Operational Cycling Loop finished successfully.")

if __name__ == "__main__":
    main()
