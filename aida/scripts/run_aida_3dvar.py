#!/usr/bin/env python3
"""
AI-DA ICOSAHEDRAL GRAPH 3D-VAR ASSIMILATION ENGINE (B-MATRIX ENHANCED)
---------------------------------------------------------------------
Assimilates conventional observations directly on a 2,562-node icosahedral mesh
(m4 level) using 3D spatially-varying B-matrix variances, full 32x32 vertical 
error correlations, Graph Message Passing, and PyTorch Autograd.
"""

import argparse
import os
import numpy as np
import torch
import xarray as xr
import zarr

STATE_VARS = ["p", "t", "u", "v", "q"]
ZARR_VAR_MAP = {
    "p": "p_icosahedral",
    "t": "t_icosahedral",
    "u": "u_icosahedral",
    "v": "v_icosahedral",
    "q": "q_icosahedral",
}
VAR_NAME_MAP_BMATRIX = {
    "p": ("p_var", "p_vert_corr"),
    "t": ("t_var", "t_vert_corr"),
    "u": ("u_var", "u_vert_corr"),
    "v": ("v_var", "v_vert_corr"),
    "q": ("q_var", "q_vert_corr"),
}


def load_zarr_background(zarr_path, time_idx=0):
    print(f"\n[1/6] Loading Icosahedral Background from Zarr: '{zarr_path}' (time index: {time_idx})")
    z_root = zarr.open(zarr_path, mode="r")

    heights = np.array(z_root["height"][:], dtype=np.float32)
    lats = np.array(z_root["latitude"][:], dtype=np.float32)
    lons = np.array(z_root["longitude"][:], dtype=np.float32)

    lons = np.where(lons > 180, lons - 360, lons)

    xb_dict = {}
    for var in STATE_VARS:
        zarr_name = ZARR_VAR_MAP[var]
        xb_dict[var] = np.array(z_root[zarr_name][time_idx, :, :], dtype=np.float32)

    n_levels, n_nodes = xb_dict["t"].shape
    print(f"  -> Background state loaded: {n_levels} vertical levels, {n_nodes} graph mesh nodes.")

    return xb_dict, lats, lons, heights


def load_graph_structure(edge_index_path):
    print(f"[2/6] Loading Graph Topology Edges from: '{edge_index_path}'")
    edge_index = torch.load(edge_index_path, map_location="cpu").long()
    print(f"  -> Graph Edges: {edge_index.shape[1]} edges across {edge_index.max().item() + 1} nodes.")
    return edge_index


def load_bmatrix_structures(bmatrix_path, mesh_lats, mesh_lons):
    """
    Interpolates 3D variances to icosahedral mesh and computes matrix square root
    (B_v^{1/2}) of 32x32 vertical error correlation matrices via Eigen decomposition.
    """
    print(f"[3/6] Ingesting B-Matrix Variances and Vertical Correlations: '{bmatrix_path}'")
    ds_b = xr.open_dataset(bmatrix_path)

    b_lons = ds_b["longitude"].values
    b_lons = np.where(b_lons > 180, b_lons - 360, b_lons)
    ds_b = ds_b.assign_coords(longitude=b_lons).sortby("longitude")

    sigma_b_3d = {}
    b_vert_half = {}

    for var in STATE_VARS:
        var_name, corr_name = VAR_NAME_MAP_BMATRIX[var]

        # 1. Spatially Varying Standard Deviation Vector \sigma_b(32, 2562)
        var_data = ds_b[var_name]
        interp_var = var_data.interp(
            latitude=xr.DataArray(mesh_lats, dims="node"),
            longitude=xr.DataArray(mesh_lons, dims="node"),
            method="linear"
        ).values
        interp_var = np.nan_to_num(interp_var, nan=1.0)
        variance = np.maximum(interp_var, 1e-8)
        
        # Convert variance to standard deviation and scale by alpha factor if defined
        alpha = float(ds_b.attrs.get("scaling_factor_alpha", 1.0))
        sigma = np.sqrt(variance) * alpha
        sigma_b_3d[var] = torch.tensor(sigma, dtype=torch.float32)

        # 2. Vertical Error Correlation Matrix C_v (32, 32) -> Matrix Sqrt C_v^{1/2}
        C_v = ds_b[corr_name].values
        C_v = np.nan_to_num(C_v, nan=0.0)
        np.fill_diagonal(C_v, 1.0)

        # Symmetric Eigen Decomposition: C_v = V * Lambda * V^T => C_v^{1/2} = V * sqrt(Lambda) * V^T
        evals, evecs = np.linalg.eigh(C_v)
        evals = np.maximum(evals, 1e-6)
        C_v_half = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

        b_vert_half[var] = torch.tensor(C_v_half, dtype=torch.float32)

    print("  -> B-Matrix successfully loaded & mapped to 3D grid.")
    return sigma_b_3d, b_vert_half


def apply_graph_laplacian_smoothing(tensor_3d, edge_index, alpha=0.35, iterations=3):
    """Applies horizontal message passing smoothing along mesh edges."""
    src, dst = edge_index[0], edge_index[1]
    n_nodes = tensor_3d.shape[1]

    out = tensor_3d.clone()
    for _ in range(iterations):
        msg = torch.zeros_like(out)
        msg.index_add_(1, dst, out[:, src])

        deg = torch.zeros(n_nodes, device=tensor_3d.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = torch.clamp(deg, min=1.0)

        msg = msg / deg.unsqueeze(0)
        out = (1.0 - alpha) * out + alpha * msg

    return out


def q_to_td_torch(q_tensor, p_tensor_pa):
    """Computes dewpoint temperature Td (K) from specific humidity q (kg/kg) and pressure p (Pa)."""
    p_hpa = torch.clamp(p_tensor_pa / 100.0, min=1.0)
    e = (q_tensor * p_hpa) / (0.622 + 0.378 * q_tensor)
    e = torch.clamp(e, min=0.001)
    
    log_e = torch.log(e / 6.1078)
    td_c = (237.3 * log_e) / (17.27 - log_e)
    return td_c + 273.15


def ingest_observations(conv_path, mesh_lats, mesh_lons, heights):
    print(f"[4/6] Ingesting & Mapping Observations: '{conv_path}'")
    try:
        ds_conv = xr.open_dataset(conv_path)
        obs_vars = ds_conv["variable"].values
        obs_vals = ds_conv["observation_value"].values.astype(np.float32)
        obs_errs = ds_conv["observation_error"].values.astype(np.float32)
        obs_lats = ds_conv["latitude"].values.astype(np.float32)
        obs_lons = ds_conv["longitude"].values.astype(np.float32)
        obs_z = ds_conv["z"].values.astype(np.float32)

        obs_lons = np.where(obs_lons > 180, obs_lons - 360, obs_lons)

        mesh_lats_rad = np.radians(mesh_lats)
        mesh_lons_rad = np.radians(mesh_lons)

        node_indices = []
        k_indices = []

        for o_lat, o_lon, o_z in zip(obs_lats, obs_lons, obs_z):
            dlat = mesh_lats_rad - np.radians(o_lat)
            dlon = mesh_lons_rad - np.radians(o_lon)
            a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(o_lat)) * np.cos(mesh_lats_rad) * np.sin(dlon / 2.0)**2
            c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

            nearest_node = np.argmin(c)
            nearest_k = np.clip(np.searchsorted(heights, o_z), 0, len(heights) - 1)

            node_indices.append(nearest_node)
            k_indices.append(nearest_k)

        accepted_obs = ["p", "t", "td", "u", "v"]
        valid_mask = np.isin(obs_vars, accepted_obs) & ~np.isnan(obs_vals) & (obs_errs > 0)

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


def execute_3dvar_analysis(zarr_path, bmatrix_path, edge_index_path, conv_path, output_path, maxiter=30):
    xb_dict, lats, lons, heights = load_zarr_background(zarr_path, time_idx=0)
    edge_index = load_graph_structure(edge_index_path)
    sigma_b_3d, b_vert_half = load_bmatrix_structures(bmatrix_path, lats, lons)

    obs_data = None
    if conv_path and os.path.exists(conv_path):
        obs_data = ingest_observations(conv_path, lats, lons, heights)

    print(f"\n[5/6] Running B-Matrix Graph-3D-Var Optimization via PyTorch Autograd...")

    n_levels, n_nodes = xb_dict["t"].shape
    n_grid = n_levels * n_nodes

    xb_tensors = {v: torch.tensor(xb_dict[v], dtype=torch.float32) for v in STATE_VARS}

    # Control vector v initialized to zero
    v_vec = torch.zeros(len(STATE_VARS) * n_grid, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.LBFGS([v_vec], max_iter=maxiter, lr=1.0, history_size=10, line_search_fn="strong_wolfe")

    step_counter = [0]

    def closure():
        optimizer.zero_grad()
        step_counter[0] += 1

        # Background Cost J_b = 0.5 * ||v||^2
        J_b = 0.5 * torch.sum(v_vec**2)

        # Full 3D State Increment: dx = \sigma_{b,3D} \cdot (C_v^{1/2} \cdot GraphSmoothing(v))
        dx_dict = {}
        for idx, var in enumerate(STATE_VARS):
            v_var = v_vec[idx * n_grid : (idx + 1) * n_grid].view(n_levels, n_nodes)

            # Step A: Horizontal Graph Laplacian propagation
            v_smoothed = apply_graph_laplacian_smoothing(v_var, edge_index)

            # Step B: Vertical correlation propagation (C_v^{1/2} @ v_smoothed)
            v_vert = torch.matmul(b_vert_half[var], v_smoothed)

            # Step C: Scale by 3D spatial standard deviations \sigma_b(32, 2562)
            dx_dict[var] = v_vert * sigma_b_3d[var]

        # Reconstructed State Vector
        x_state = {var: xb_tensors[var] + dx_dict[var] for var in STATE_VARS}

        # Observation Cost J_o
        J_o = torch.tensor(0.0, dtype=torch.float32)
        if obs_data is not None and len(obs_data["value"]) > 0:
            obs_vals = obs_data["value"]
            obs_errs = obs_data["error"]
            obs_vars = obs_data["variable"]
            obs_nodes = obs_data["node"]
            obs_ks = obs_data["k"]

            H_x = torch.zeros_like(obs_vals)

            for var in ["p", "t", "u", "v"]:
                mask = (obs_vars == var)
                # if torch.any(mask):
                if mask.any():
                    k_m = obs_ks[mask]
                    node_m = obs_nodes[mask]
                    H_x[mask] = x_state[var][k_m, node_m]

            # Dewpoint observation operator Td = H_td(q, p)
            mask_td = (obs_vars == "td")
            # if torch.any(mask_td):
            if mask_td.any():
                k_m = obs_ks[mask_td]
                node_m = obs_nodes[mask_td]
                q_state = x_state["q"][k_m, node_m]
                p_state = x_state["p"][k_m, node_m]
                H_x[mask_td] = q_to_td_torch(q_state, p_state)

            residual = obs_vals - H_x
            J_o = 0.5 * torch.sum((residual / obs_errs)**2)

        J_total = J_b + J_o
        J_total.backward()

        if step_counter[0] % 5 == 0 or step_counter[0] == 1:
            print(f"  Step {step_counter[0]:02d} | Cost J: {J_total.item():.4f} (J_b: {J_b.item():.4f}, J_o: {J_o.item():.4f})")

        return J_total

    optimizer.step(closure)

    # Extract final analysis states x_a
    xa_dict = {}
    with torch.no_grad():
        for idx, var in enumerate(STATE_VARS):
            v_var = v_vec[idx * n_grid : (idx + 1) * n_grid].view(n_levels, n_nodes)
            v_smoothed = apply_graph_laplacian_smoothing(v_var, edge_index)
            v_vert = torch.matmul(b_vert_half[var], v_smoothed)
            dx_opt = v_vert * sigma_b_3d[var]
            xa_dict[var] = (xb_tensors[var] + dx_opt).cpu().numpy()

    # Export NetCDF Analysis Output
    print(f"\n[6/6] Exporting Analysis to NetCDF: '{output_path}'")
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
            "title": "AI-DA 5-Variable Icosahedral Graph 3D-Var Analysis",
            "institution": "Anemoi / Starviewer AI-DA System",
            "b_matrix_used": "Analysis-Based Variance + Vertical Correlation Matrix",
            "mesh_level": "m4 (2562 nodes, 15360 edges)",
            "iterations": str(step_counter[0]),
        },
    )

    out_ds.to_netcdf(output_path)
    print("==================================================================")
    print("AI-DA B-Matrix Graph 3D-Var Assimilation Completed Successfully!")
    print("==================================================================")
    return xa_dict


def main():
    parser = argparse.ArgumentParser(description="Run AI-DA Graph-based 3D-Var Engine with 3D B-Matrix")
    parser.add_argument("--zarr", type=str, default="../data/icosahedral_2023.zarr")
    parser.add_argument("--bmatrix", type=str, default="../bmatrix/bmatrix_from_gfs_analysis.nc")
    parser.add_argument("--edges", type=str, default="../data/graph/edge_index_m4.pt")
    parser.add_argument("--conv", type=str, default="../data/conv_2023/conv.20230306.t18z.nc")
    parser.add_argument("--output", type=str, default="aida_icosahedral_analysis.nc")
    parser.add_argument("--maxiter", type=int, default=30)

    args = parser.parse_args()

    execute_3dvar_analysis(
        zarr_path=args.zarr,
        bmatrix_path=args.bmatrix,
        edge_index_path=args.edges,
        conv_path=args.conv,
        output_path=args.output,
        maxiter=args.maxiter
    )


if __name__ == "__main__":
    main()

