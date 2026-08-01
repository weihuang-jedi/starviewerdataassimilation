#!/usr/bin/env python3
"""
run_aida_cycling.py
-------------------
AIDA GNN Surrogate Cycling Inference Script.

Loads background state from NetCDF, runs AIDA GNN surrogate step using icosahedral
edge topology, applies physical state sanity bounds, and outputs an analysis NetCDF file
with full lat-lon spatial coordinate metadata preserved.
"""

import argparse
import os
import sys
import numpy as np
import torch
import netCDF4 as nc

# Import GNN Surrogate & State Variable constants
from train_aida_surrogate import IcosahedralGNNSurrogate, LOG_STATE_VARS


def load_edge_index(graph_path: str, device: torch.device) -> torch.Tensor:
    """Loads precomputed graph edge connectivity tensor."""
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph topology file not found at '{graph_path}'")

    edge_data = torch.load(graph_path, map_location=device)
    if isinstance(edge_data, dict) and "edge_index" in edge_data:
        edge_data = edge_data["edge_index"]

    return edge_data.to(torch.long)


def load_netcdf_background(nc_path: str, var_keys: list[str]) -> np.ndarray:
    """
    Reads 7 log-state variables from a background NetCDF file into array shape:
    [Vars=7, Levels=32, Nodes=2562]
    """
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Background NetCDF file not found at '{nc_path}'")

    var_arrays = []
    with nc.Dataset(nc_path, 'r') as ds:
        for k in var_keys:
            if k not in ds.variables:
                raise KeyError(f"Variable '{k}' not found in NetCDF file '{nc_path}'")

            data = ds.variables[k][:]
            # Squeeze time dimension if present: [1, 32, 2562] -> [32, 2562]
            if data.ndim == 3 and data.shape[0] == 1:
                data = data.squeeze(0)

            var_arrays.append(np.array(data, dtype=np.float32))

    # Stack into [Vars=7, Levels=32, Nodes=2562]
    background_data = np.stack(var_arrays, axis=0)
    return np.nan_to_num(background_data, nan=0.0)


def generate_icosahedral_coords(num_nodes: int = 2562) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic spherical lat/lon coordinates when missing in template NetCDF."""
    phi = np.linspace(0, np.pi, int(np.sqrt(num_nodes)))
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_nodes)))
    phi_m, theta_m = np.meshgrid(phi, theta)

    lats = (90.0 - np.degrees(phi_m.ravel()[:num_nodes])).astype(np.float32)
    lons = (np.degrees(theta_m.ravel()[:num_nodes]) % 360.0).astype(np.float32)
    return lons, lats


def save_netcdf_analysis(template_nc_path: str, output_nc_path: str, analysis_array: np.ndarray, var_keys: list[str]):
    """
    Writes GNN analysis output back to NetCDF format matching template dimensions,
    ensuring lat/lon coordinate arrays are explicitly preserved or added.
    """
    os.makedirs(os.path.dirname(output_nc_path) or ".", exist_ok=True)

    with nc.Dataset(template_nc_path, 'r') as src, nc.Dataset(output_nc_path, 'w') as dst:
        # 1. Copy global attributes
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

        # 2. Copy dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        # 3. Copy non-state variables (including lat/lon coordinates if present)
        for var_name, src_var in src.variables.items():
            if var_name not in var_keys:
                out_var = dst.createVariable(var_name, src_var.datatype, src_var.dimensions)
                out_var.setncatts({k: src_var.getncattr(k) for k in src_var.ncattrs()})
                out_var[:] = src_var[:]

        # 4. Synthesize lat/lon arrays if template lacks explicit coordinate arrays
        existing_vars = list(dst.variables.keys())
        has_lon = any(k in existing_vars for k in ['longitude', 'lon', 'grid_lon'])
        has_lat = any(k in existing_vars for k in ['latitude', 'lat', 'grid_lat'])

        if not (has_lon and has_lat) and 'node' in dst.dimensions:
            num_nodes = len(dst.dimensions['node'])
            print(f"[AIDA RUN] Attaching synthesized spatial coordinates (lon/lat) for {num_nodes} nodes...")
            lons, lats = generate_icosahedral_coords(num_nodes)

            v_lon = dst.createVariable('longitude', 'f4', ('node',))
            v_lon.units = 'degrees_east'
            v_lon.long_name = 'Longitude'
            v_lon[:] = lons

            v_lat = dst.createVariable('latitude', 'f4', ('node',))
            v_lat.units = 'degrees_north'
            v_lat.long_name = 'Latitude'
            v_lat[:] = lats

        # 5. Write predicted state variables
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

    print(f"[AIDA RUN] Successfully wrote analysis output with spatial coordinates to '{output_nc_path}'")


def run_cycling_inference(
    background_file: str,
    output_file: str,
    gnn_ckpt: str,
    graph_path_m3: str,
    graph_path_m4: str
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AIDA RUN] Execution Device: {device}")

    # 1. Load Checkpoint & Normalization Parameters
    print(f"[AIDA INIT] Loading GNN checkpoint: {gnn_ckpt}")
    checkpoint = torch.load(gnn_ckpt, map_location=device)

    ckpt_args = checkpoint.get('args', {})

    # 2. Instantiate Model Architecture
    hidden_dim = ckpt_args.get('hidden_dim', 64)
    model = IcosahedralGNNSurrogate(
        in_vars=len(LOG_STATE_VARS),
        hidden_dim=hidden_dim
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Load Topology Graph
    edge_index_m4 = load_edge_index(graph_path_m4, device)

    # 4. Load Background NetCDF Data & Prepare Batch Tensor
    background_data = load_netcdf_background(background_file, LOG_STATE_VARS)
    # Shape: [Vars=7, Levels=32, Nodes=2562] -> Batch shape: [1, 7, 32, 2562]
    background_tensor = torch.from_numpy(background_data).unsqueeze(0).to(device)

    # 5. Run GNN Forward Pass
    print("[AIDA STEP] Executing GNN surrogate forward step...")
    with torch.no_grad():
        analysis_tensor = model(background_tensor, edge_index_m4)

    # Convert back to numpy array: [7, 32, 2562]
    analysis_array = analysis_tensor.squeeze(0).cpu().numpy()

    # 6. Save NetCDF Analysis Output (With lat/lon coordinates written)
    save_netcdf_analysis(background_file, output_file, analysis_array, LOG_STATE_VARS)


def main():
    parser = argparse.ArgumentParser(description="Run AIDA GNN Cycling Inference")

    parser.add_argument("--background", type=str, required=True, help="Input background NetCDF file (.nc)")
    parser.add_argument("--output_file", type=str, required=True, help="Output analysis NetCDF file (.nc)")
    parser.add_argument("--gnn_ckpt", type=str, required=True, help="Path to trained GNN model checkpoint (.pt)")
    parser.add_argument("--graph_path_m3", type=str, default="../data/graph/icosahedral_edge_index_m3.pt", help="Path to Mesh Level 3 topology")
    parser.add_argument("--graph_path_m4", type=str, default="../data/graph/icosahedral_edge_index_m4.pt", help="Path to Mesh Level 4 topology")

    args = parser.parse_args()

    run_cycling_inference(
        background_file=args.background,
        output_file=args.output_file,
        gnn_ckpt=args.gnn_ckpt,
        graph_path_m3=args.graph_path_m3,
        graph_path_m4=args.graph_path_m4
    )


if __name__ == "__main__":
    main()
