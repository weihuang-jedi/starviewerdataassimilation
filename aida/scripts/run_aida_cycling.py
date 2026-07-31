#!/usr/bin/env python3
"""
run_aida_cycling.py
-------------------
Operational AI-DA Cycling Loop script.
Integrates PINN-trained GNN log-state surrogate forecasts with 3D-Var assimilation 
and the MultiMeshHierarchicalDecoder across operational cycles.
"""

import os
import sys
import glob
import argparse
import numpy as np
import xarray as xr
import torch

# Ensure the root project folder is in sys.path for direct module discovery
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_aida_surrogate import (
    IcosahedralGNNSurrogate,
    LOG_STATE_VARS,
    generate_or_load_edge_index
)
from models.hierarchical_decoder import MultiMeshHierarchicalDecoder

# Physical Gas Constant for Dry Air
R_DRY = 287.058


# ==========================================
# 1. MESH 3 OBSERVATION INNOVATION GENERATOR
# ==========================================
def generate_mesh3_innovations(batch_size, num_vars, levels, num_m3_nodes, device):
    """
    Simulates observational innovations ingested at Mesh 3 resolution (e.g. 642 nodes).
    In production, this interface connects directly to NNJA-AI / PyArrow conventional obs.
    """
    # Scale innovations appropriately for log-space variables vs velocity components
    scale = torch.tensor([0.002, 0.01, 0.01, 0.001, 0.005, 0.002, 0.002], device=device).view(1, num_vars, 1, 1)
    raw_noise = torch.randn(batch_size, num_vars, levels, num_m3_nodes, device=device)
    return raw_noise * scale


# ==========================================
# 2. PHYSICAL CONSTRAINTS IN LOG SPACE
# ==========================================
def apply_logstate_physical_constraints(analysis_tensor, var_names):
    """
    Applies physical bounds directly in log-state space.
    """
    # Index locations
    q_idx = var_names.index('q_icosahedral') if 'q_icosahedral' in var_names else -1
    ln_t_idx = var_names.index('ln_t_icosahedral') if 'ln_t_icosahedral' in var_names else -1
    ln_p_idx = var_names.index('ln_p_icosahedral') if 'ln_p_icosahedral' in var_names else -1

    # Moisture positivity constraint (q >= 1e-8 kg/kg)
    if q_idx != -1:
        analysis_tensor[:, q_idx, :, :] = torch.clamp(analysis_tensor[:, q_idx, :, :], min=1e-8)

    # Temperature floor bound (T >= 100 K -> ln_T >= 4.605)
    if ln_t_idx != -1:
        analysis_tensor[:, ln_t_idx, :, :] = torch.clamp(analysis_tensor[:, ln_t_idx, :, :], min=4.605)

    # Pressure floor bound (p >= 1.0 Pa -> ln_p >= 0.0)
    if ln_p_idx != -1:
        analysis_tensor[:, ln_p_idx, :, :] = torch.clamp(analysis_tensor[:, ln_p_idx, :, :], min=0.0)

    return analysis_tensor


# ==========================================
# 3. MAIN CYCLING LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="AIDA Operational Cycling with MultiMesh Hierarchical Decoder.")
    parser.add_argument("--gnn_ckpt", type=str, default="checkpoints/aida_gnn_surrogate_logstate.pt", help="Path to log-state checkpoint")
    parser.add_argument("--graph_path_m4", type=str, default="../data/graph/icosahedral_edge_index_m4.pt", help="Mesh 4 edge topology")
    parser.add_argument("--graph_path_m3", type=str, default="../data/graph/icosahedral_edge_index_m3.pt", help="Mesh 3 edge topology")
    parser.add_argument("--data_dir", type=str, default="../data/nc", help="Input background NetCDF directory")
    parser.add_argument("--obs_dir", type=str, default="../data/obs", help="Observation directory")
    parser.add_argument("--output_dir", type=str, default="output/cycling_logstate", help="Output directory")
    parser.add_argument("--cycles", type=int, default=4, help="Number of 6h cycling iterations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Checkpoint and Metadata
    print(f"[AIDA GNN] Loading log-state checkpoint from '{args.gnn_ckpt}'...")
    checkpoint = torch.load(args.gnn_ckpt, map_location=device)

    var_names = checkpoint.get('var_names', LOG_STATE_VARS)
    stats = checkpoint['stats']
    num_vars = len(var_names)

    # Load Mesh Topologies
    edge_index_m4 = generate_or_load_edge_index(num_nodes=2562, edge_file=args.graph_path_m4).to(device)
    edge_index_m3 = generate_or_load_edge_index(num_nodes=642, edge_file=args.graph_path_m3).to(device)

    # Normalize Tensors (Shape: 1, Num_Vars, 1, 1)
    mean_list = [stats[v]['mean'] for v in var_names]
    std_list  = [stats[v]['std'] for v in var_names]

    var_mean = torch.tensor(mean_list, dtype=torch.float32, device=device).view(1, num_vars, 1, 1)
    var_std  = torch.tensor(std_list, dtype=torch.float32, device=device).view(1, num_vars, 1, 1)

    # Instantiate Forward Surrogate Model
    gnn_model = IcosahedralGNNSurrogate(
        edge_index=edge_index_m4,
        in_vars=num_vars,
        levels=32
    ).to(device)
    gnn_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    gnn_model.eval()

    # Instantiate Hierarchical Multi-Mesh Decoder (Mesh 3 -> Mesh 4)
    decoder_model = MultiMeshHierarchicalDecoder(
        edge_index_m3=edge_index_m3,
        edge_index_m4=edge_index_m4,
        num_m3_nodes=642,
        num_m4_nodes=2562,
        in_vars=num_vars,
        levels=32
    ).to(device)
    decoder_model.eval()

    # 2. Ingest Baseline Background NetCDF File
    input_files = sorted(glob.glob(os.path.join(args.data_dir, "*.nc")))
    if not input_files:
        raise FileNotFoundError(f"No NetCDF files found in directory: {args.data_dir}")

    current_nc_path = input_files[0]
    print(f"[CYCLE 0] Baseline background initialized from '{current_nc_path}'...")
    ds_curr = xr.open_dataset(current_nc_path)

    # Verify all log variables exist in NetCDF
    for v in var_names:
        if v not in ds_curr.data_vars:
            raise KeyError(f"Variable '{v}' missing from input NetCDF: {current_nc_path}")

    # Stack state: shape (7, 32, 2562)
    curr_state_np = np.stack([ds_curr[v].values for v in var_names], axis=0)
    curr_state = torch.tensor(curr_state_np, dtype=torch.float32, device=device).unsqueeze(0) # (1, 7, 32, 2562)

    # 3. Operational Cycling Execution
    for cycle in range(1, args.cycles + 1):
        print(f"\n========================================================")
        print(f"[AIDA CYCLING] Cycle {cycle}/{args.cycles}")
        print(f"========================================================")

        # Step A: Forward Forecast Pass (Mesh 4)
        x_in_norm = (curr_state - var_mean) / var_std

        with torch.no_grad():
            pred_norm = gnn_model(x_in_norm)
            bg_state = (pred_norm * var_std) + var_mean

        print(f"[FORECAST] GNN 6h scale-invariant forecast step complete.")

        # Step B: Ingest Observations at Mesh 3 & Decode Analysis Increments to Mesh 4
        obs_innov_m3 = generate_mesh3_innovations(
            batch_size=1,
            num_vars=num_vars,
            levels=32,
            num_m3_nodes=642,
            device=device
        )

        with torch.no_grad():
            an_state = decoder_model(bg_state, obs_innov_m3)
            an_state = apply_logstate_physical_constraints(an_state, var_names)

        print(f"[HIERARCHICAL DECODER] Ingested Mesh 3 observations and updated Mesh 4 analysis state.")

        # Step C: Export NetCDF Analysis File (Log-State + Physical Variables)
        out_nc_path = os.path.join(args.output_dir, f"aida_analysis_cycle_{cycle:02d}.nc")
        ds_out = ds_curr.copy(deep=True)

        an_state_np = an_state.squeeze(0).cpu().numpy()

        for idx, real_var_name in enumerate(var_names):
            ds_out[real_var_name].values = an_state_np[idx]

        # Compute physical temperature and pressure fields for easy visualization
        if 'ln_t_icosahedral' in var_names:
            ds_out['t_physical'] = np.exp(ds_out['ln_t_icosahedral'])
        if 'ln_p_icosahedral' in var_names:
            ds_out['p_physical'] = np.exp(ds_out['ln_p_icosahedral'])

        ds_out.to_netcdf(out_nc_path)
        print(f"[OUTPUT] Cycle analysis output exported to '{out_nc_path}'")

        # Update current state for the next cycle
        curr_state = an_state

    ds_curr.close()
    print("\n[COMPLETE] AI-DA Operational Cycling Loop with MultiMesh Decoder finished successfully.")

if __name__ == "__main__":
    main()
