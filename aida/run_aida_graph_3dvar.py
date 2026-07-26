#!/usr/bin/env python3
"""
AI-DA ICOSAHEDRAL GRAPH 3D-VAR ASSIMILATION ENGINE
--------------------------------------------------
Assimilates conventional/radiance observations directly on a 2,562-node
icosahedral mesh (m4 level) using Graph Message Passing for B-Matrix covariances
and PyTorch Autograd for Cost Function Minimization.
"""

import argparse
from datetime import datetime
import numpy as np
import torch
import xarray as xr
import zarr

# State variables on icosahedral mesh
STATE_VARS = ["p", "t", "u", "v", "q"]
ZARR_VAR_MAP = {
    "p": "p_icosahedral",
    "t": "t_icosahedral",
    "u": "u_icosahedral",
    "v": "v_icosahedral",
    "q": "q_icosahedral",
}

# Background Error Standard Deviations (\sigma_b)
SIGMA_B = {
    "p": 100.0,    # Pressure (Pa)
    "t": 1.5,      # Temperature (K)
    "u": 3.0,      # Zonal Wind (m/s)
    "v": 3.0,      # Meridional Wind (m/s)
    "q": 0.001,    # Specific Humidity (kg/kg)
}


def load_zarr_background(zarr_path, time_idx=0):
    print(f"\n[1/5] Loading Icosahedral Background from Zarr: '{zarr_path}' (time index: {time_idx})")
    z_root = zarr.open(zarr_path, mode="r")

    heights = np.array(z_root["height"][:], dtype=np.float32)
    lats = np.array(z_root["latitude"][:], dtype=np.float32)
    lons = np.array(z_root["longitude"][:], dtype=np.float32)

    # Normalize longitudes to [-180, 180]
    lons = np.where(lons > 180, lons - 360, lons)

    xb_dict = {}
    for var in STATE_VARS:
        zarr_name = ZARR_VAR_MAP[var]
        # Data shape: (32, 2562)
        xb_dict[var] = np.array(z_root[zarr_name][time_idx, :, :], dtype=np.float32)

    n_levels, n_nodes = xb_dict["t"].shape
    print(f"  -> Successfully loaded background state: {n_levels} vertical levels, {n_nodes} graph mesh nodes.")

    return xb_dict, lats, lons, heights


def load_graph_structure(edge_index_path):
    print(f"[2/5] Loading Graph Topology Edges from: '{edge_index_path}'")
    edge_index = torch.load(edge_index_path, map_location="cpu").long()
    print(f"  -> Graph Edges: {edge_index.shape[1]} edges across {edge_index.max().item() + 1} nodes.")
    return edge_index


def apply_graph_laplacian_smoothing(tensor_3d, edge_index, alpha=0.35, iterations=3):
    """
    Applies Graph Message Passing Smoothing along icosahedral mesh edges.
    Propagates observation innovations across adjacent mesh nodes.
    tensor_3d: (32, 2562)
    """
    src, dst = edge_index[0], edge_index[1]
    n_nodes = tensor_3d.shape[1]

    out = tensor_3d.clone()
    for _ in range(iterations):
        # Gather node features along edges and aggregate via mean
        msg = torch.zeros_like(out)
        msg.index_add_(1, dst, out[:, src])

        # Degree normalization
        deg = torch.zeros(n_nodes, device=tensor_3d.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = torch.clamp(deg, min=1.0)

        msg = msg / deg.unsqueeze(0)
        out = (1.0 - alpha) * out + alpha * msg

    return out


def ingest_observations(conv_path, mesh_lats, mesh_lons, heights):
    print(f"[3/5] Ingesting & Mapping Conventional Obs to Icosahedral Mesh: '{conv_path}'")
    try:
        ds_conv = xr.open_dataset(conv_path)
        obs_vars = ds_conv["variable"].values
        obs_vals = ds_conv["observation_value"].values.astype(np.float32)
        obs_errs = ds_conv["observation_error"].values.astype(np.float32)
        obs_lats = ds_conv["latitude"].values.astype(np.float32)
        obs_lons = ds_conv["longitude"].values.astype(np.float32)
        obs_lvls_hpa = ds_conv["level"].values.astype(np.float32)

        obs_lons = np.where(obs_lons > 180, obs_lons - 360, obs_lons)
        obs_z = 7000.0 * np.log(1013.25 / np.maximum(obs_lvls_hpa, 0.1))

        # Build Spherical KD-Tree / Haversine Nearest Node Mapping
        mesh_lats_rad = np.radians(mesh_lats)
        mesh_lons_rad = np.radians(mesh_lons)

        node_indices = []
        k_indices = []

        for o_lat, o_lon, o_z in zip(obs_lats, obs_lons, obs_z):
            # Haversine distance to all 2,562 mesh nodes
            dlat = mesh_lats_rad - np.radians(o_lat)
            dlon = mesh_lons_rad - np.radians(o_lon)
            a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(o_lat)) * np.cos(mesh_lats_rad) * np.sin(dlon / 2.0)**2
            c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

            nearest_node = np.argmin(c)
            nearest_k = np.clip(np.searchsorted(heights, o_z), 0, len(heights) - 1)

            node_indices.append(nearest_node)
            k_indices.append(nearest_k)

        valid_mask = np.isin(obs_vars, STATE_VARS) & ~np.isnan(obs_vals) & (obs_errs > 0)

        obs_data = {
            "variable": obs_vars[valid_mask],
            "value": torch.tensor(obs_vals[valid_mask], dtype=torch.float32),
            "error": torch.tensor(obs_errs[valid_mask], dtype=torch.float32),
            "node": torch.tensor(np.array(node_indices)[valid_mask], dtype=torch.long),
            "k": torch.tensor(np.array(k_indices)[valid_mask], dtype=torch.long),
        }

        print(f"  -> Retained {len(obs_data['value'])} valid observations mapped to Graph Nodes.")
        return obs_data

    except Exception as e:
        print(f"  -> Observation ingestion failed ({e}). Proceeding without obs.")
        return None


def run_graph_3dvar(xb_dict, edge_index, obs_data, maxiter=30):
    print(f"\n[4/5] Running Graph-based 3D-Var Optimization via PyTorch Autograd...")

    n_levels, n_nodes = xb_dict["t"].shape
    n_grid = n_levels * n_nodes

    # Convert background to PyTorch Tensors
    xb_tensors = {v: torch.tensor(xb_dict[v], dtype=torch.float32) for v in STATE_VARS}

    # Initialize control variable v = 0
    v_vec = torch.zeros(len(STATE_VARS) * n_grid, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.LBFGS([v_vec], max_iter=maxiter, lr=1.0, history_size=10, line_search_fn="strong_wolfe")

    step_counter = [0]

    def closure():
        optimizer.zero_grad()
        step_counter[0] += 1

        # Background Cost J_b = 0.5 * ||v||^2
        J_b = 0.5 * torch.sum(v_vec**2)

        # Reconstruct state increment fields delta_x = B^{1/2} v
        dx_dict = {}
        for idx, var in enumerate(STATE_VARS):
            v_var = v_vec[idx * n_grid : (idx + 1) * n_grid].view(n_levels, n_nodes)
            # Apply Graph B^{1/2}: Scale by sigma_b and Graph Laplacian smoothing
            smoothed_v = apply_graph_laplacian_smoothing(v_var, edge_index)
            dx_dict[var] = smoothed_v * SIGMA_B[var]

        # Observation Cost J_o
        J_o = torch.tensor(0.0, dtype=torch.float32)
        if obs_data is not None and len(obs_data["value"]) > 0:
            obs_vals = obs_data["value"]
            obs_errs = obs_data["error"]
            obs_vars = obs_data["variable"]
            obs_nodes = obs_data["node"]
            obs_ks = obs_data["k"]

            H_x = torch.zeros_like(obs_vals)

            for idx, var in enumerate(STATE_VARS):
                mask = (obs_vars == var)
                if torch.any(mask):
                    k_m = obs_ks[mask]
                    node_m = obs_nodes[mask]

                    xb_m = xb_tensors[var][k_m, node_m]
                    dx_m = dx_dict[var][k_m, node_m]

                    H_x[mask] = xb_m + dx_m

            residual = obs_vals - H_x
            J_o = 0.5 * torch.sum((residual / obs_errs)**2)

        J_total = J_b + J_o
        J_total.backward()

        if step_counter[0] % 5 == 0 or step_counter[0] == 1:
            print(f"  Step {step_counter[0]:02d} | Cost J: {J_total.item():.4f} (J_b: {J_b.item():.4f}, J_o: {J_o.item():.4f})")

        return J_total

    optimizer.step(closure)

    # Compute final analysis states xa = xb + B^{1/2} v_opt
    xa_dict = {}
    with torch.no_grad():
        for idx, var in enumerate(STATE_VARS):
            v_var = v_vec[idx * n_grid : (idx + 1) * n_grid].view(n_levels, n_nodes)
            dx_opt = apply_graph_laplacian_smoothing(v_var, edge_index) * SIGMA_B[var]
            xa_dict[var] = (xb_tensors[var] + dx_opt).cpu().numpy()

    return xa_dict, step_counter[0]


def main():
    parser = argparse.ArgumentParser(description="Run AI-DA Graph-based 3D-Var Engine on Icosahedral Mesh")
    parser.add_argument("--zarr", type=str, default="../data/icosahedral_2023.zarr", help="Path to background Zarr")
    parser.add_argument("--edges", type=str, default="../data/graph/edge_index_m4.pt", help="Path to PyTorch graph edges")
    parser.add_argument("--conv", type=str, default="conv_adpupa_NC002001.nc", help="Path to conventional observations")
    parser.add_argument("--output", type=str, default="aida_icosahedral_analysis.nc", help="Output NetCDF analysis file")
    parser.add_argument("--maxiter", type=int, default=30, help="Max optimization iterations")

    args = parser.parse_args()

    xb_dict, lats, lons, heights = load_zarr_background(args.zarr, time_idx=0)
    edge_index = load_graph_structure(args.edges)

    obs_data = None
    if args.conv:
        obs_data = ingest_observations(args.conv, lats, lons, heights)

    xa_dict, n_iters = run_graph_3dvar(xb_dict, edge_index, obs_data, maxiter=args.maxiter)

    print(f"\n[5/5] Exporting AI-DA Analysis Output to NetCDF: '{args.output}'")
    data_vars = {}
    for var in STATE_VARS:
        bg = xb_dict[var]
        anal = xa_dict[var]
        inc = anal - bg

        v_upper = var.upper()
        data_vars[f"{var}_background"] = (("height", "node"), bg, {"long_name": f"AI-DA Background ({v_upper})"})
        data_vars[f"{var}_analysis"] = (("height", "node"), anal, {"long_name": f"AI-DA Graph Analysis ({v_upper})"})
        data_vars[f"{var}_increment"] = (("height", "node"), inc, {"long_name": f"AI-DA Graph Increment ({v_upper})"})

    out_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "height": ("height", heights),
            "node": ("node", np.arange(len(lats))),
            "latitude": ("node", lats),
            "longitude": ("node", lons),
        },
        attrs={
            "title": "AI-DA 5-Variable Icosahedral Mesh Graph 3D-Var Analysis",
            "institution": "Anemoi / Starviewer AI-DA System",
            "mesh_level": "m4 (2562 nodes, 15360 edges)",
            "variables_assimilated": "p, t, u, v, q",
            "iterations": str(n_iters),
        },
    )

    out_ds.to_netcdf(args.output)
    print("==================================================================")
    print("AI-DA Graph 3D-Var Assimilation Completed Successfully!")
    print("==================================================================")


if __name__ == "__main__":
    main()
