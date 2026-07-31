#!/usr/bin/env python3
"""
Operational Cycling Driver for AIDA GNN Surrogate Model.
Executes multi-cycle forecast-assimilation updates on icosahedral unstructured grids.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import xarray as xr
import numpy as np


def load_gnn_model(checkpoint_path: str, graph_path_m4: str, device: str = "cpu") -> nn.Module:
    """
    Loads GNN checkpoint, dynamically resolving class dependencies and graph topology.
    """
    print(f"[AIDA INIT] Loading GNN checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 1. Load topology required for initialization
    print(f"[AIDA INIT] Loading Mesh Topology for model init: {graph_path_m4}")
    edge_index_m4 = torch.load(graph_path_m4, map_location=device)

    # 2. Extract state dict
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        # Dynamically import surrogate class from train script
        try:
            from scripts.train_aida_surrogate import IcosahedralGNNSurrogate
        except ImportError:
            try:
                from train_aida_surrogate import IcosahedralGNNSurrogate
            except ImportError as e:
                raise RuntimeError(
                    f"Could not import IcosahedralGNNSurrogate from train_aida_surrogate: {e}"
                )

        # Instantiate with positional or keyword argument for topology
        try:
            model = IcosahedralGNNSurrogate(edge_index=edge_index_m4)
        except TypeError:
            model = IcosahedralGNNSurrogate(edge_index_m4)

        model.load_state_dict(state_dict)
    else:
        raise TypeError(f"Unrecognized checkpoint format: {type(ckpt)}")

    model.to(device)
    model.eval()
    return model


def run_single_aida_cycle(
    bg_file: str,
    output_file: str,
    gnn_model: nn.Module,
    edge_m4: torch.Tensor,
    edge_m3: torch.Tensor,
    expected_num_vars: int = 7,
    device: str = "cpu",
):
    """
    Ingests background file, shapes 7-variable tensor, executes forward pass, 
    and saves analysis NetCDF output.
    """
    print(f"\n[AIDA RUN] Processing Background File: {bg_file}")
    ds_bg = xr.open_dataset(bg_file)

    print("[AIDA RUN] Ingesting state variables into PyTorch buffers...")
    ln_t = torch.tensor(ds_bg["ln_t_icosahedral"].values, dtype=torch.float32)
    ln_p = torch.tensor(ds_bg["ln_p_icosahedral"].values, dtype=torch.float32)
    ln_rho = torch.tensor(ds_bg["ln_rho_icosahedral"].values, dtype=torch.float32)

    # Secondary / Auxiliary variables if available in NetCDF, else fill with zeros
    aux_vars = ["u_icosahedral", "v_icosahedral", "q_icosahedral", "w_icosahedral"]
    var_list = [ln_t, ln_p, ln_rho]

    for v_name in aux_vars:
        if v_name in ds_bg.data_vars:
            var_list.append(torch.tensor(ds_bg[v_name].values, dtype=torch.float32))
        else:
            # Zero-pad missing auxiliary variables to match 7-var model requirement
            var_list.append(torch.zeros_like(ln_t))

    # Ensure feature vector matches expected input dimensions (7 variables x 32 levels = 224 channels)
    while len(var_list) < expected_num_vars:
        var_list.append(torch.zeros_like(ln_t))

    # Stack variables -> shape: (7, levels, num_nodes)
    raw_vars = torch.stack(var_list, dim=0)

    # Format 4D Batch input: (batch_size=1, num_vars=7, levels=32, num_nodes=2562)
    if raw_vars.dim() == 3:
        x_in = raw_vars.unsqueeze(0).to(device)
    elif raw_vars.dim() == 2:
        x_in = raw_vars.unsqueeze(0).unsqueeze(2).to(device)
    else:
        raise ValueError(f"Unexpected tensor shape: {raw_vars.shape}")

    print(f"[AIDA GNN] Evaluating surrogate forecast model with input shape {list(x_in.shape)}...")
    with torch.no_grad():
        output_delta = gnn_model(x_in)

    # Format outputs
    if output_delta.dim() == 4:
        output_delta = output_delta.squeeze(0)  # (7, levels, num_nodes)

    # Extract log-state updates
    delta_ln_t = output_delta[0].cpu().numpy()
    delta_ln_p = output_delta[1].cpu().numpy()
    delta_ln_rho = output_delta[2].cpu().numpy()

    # Apply predicted analysis increments
    ds_anal = ds_bg.copy(deep=True)
    ds_anal["ln_t_icosahedral"].values += delta_ln_t
    ds_anal["ln_p_icosahedral"].values += delta_ln_p
    ds_anal["ln_rho_icosahedral"].values += delta_ln_rho

    # Add cycle tracking attributes
    ds_anal.attrs["aida_status"] = "ANALYSIS_CYCLE_COMPLETE"
    ds_anal.attrs["aida_gnn_applied"] = "TRUE"

    # Export analysis file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    ds_anal.to_netcdf(output_file)
    print(f"[AIDA SUCCESS] Exported analysis file: {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="AIDA GNN Surrogate Cycling Loop Driver")
    parser.add_argument("--background", type=str, required=True, help="Path to input NetCDF background file")
    parser.add_argument("--output_file", type=str, required=True, help="Path for output NetCDF analysis file")
    parser.add_argument("--gnn_ckpt", type=str, required=True, help="Path to GNN surrogate PyTorch checkpoint")
    parser.add_argument("--graph_path_m4", type=str, required=True, help="Path to mesh M4 topology edge tensor")
    parser.add_argument("--graph_path_m3", type=str, required=True, help="Path to mesh M3 topology edge tensor")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute GNN model (cpu/cuda)")
    args = parser.parse_args()

    # 1. Load topological graphs
    print(f"[AIDA INIT] Loading Mesh Topology M4: {args.graph_path_m4}")
    edge_m4 = torch.load(args.graph_path_m4, map_location=args.device)
    print(f"[AIDA INIT] Loading Mesh Topology M3: {args.graph_path_m3}")
    edge_m3 = torch.load(args.graph_path_m3, map_location=args.device)

    # 2. Load surrogate model
    gnn_model = load_gnn_model(args.gnn_ckpt, args.graph_path_m4, device=args.device)

    # 3. Execute cycling step
    run_single_aida_cycle(
        bg_file=args.background,
        output_file=args.output_file,
        gnn_model=gnn_model,
        edge_m4=edge_m4,
        edge_m3=edge_m3,
        expected_num_vars=7,
        device=args.device,
    )


if __name__ == "__main__":
    main()
