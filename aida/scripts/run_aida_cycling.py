#!/usr/bin/env python3
"""
scripts/run_aida_cycling.py
---------------------------
Operational AI-DA Cycling Script.
Orchestrates GNN forecast steps and 3D-Var assimilation steps over a multi-cycle period.
"""

import argparse
import os
import subprocess
import sys
import datetime
import torch
import xarray as xr
import zarr
import numpy as np

# Import model definition from training script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_aida_surrogate import IcosahedralGraphGNN, STATE_VARS, ZARR_VAR_MAP


def run_gnn_forecast(model, x_init_tensor, device):
    """
    Runs a 6-hour forecast step using the trained GNN surrogate model.
    x_init_tensor shape: (1, 5, 32, 2562)
    Returns predicted background tensor of shape: (1, 5, 32, 2562)
    """
    model.eval()
    with torch.no_grad():
        x_init_tensor = x_init_tensor.to(device)
        x_bg_tensor = model(x_init_tensor)
    return x_bg_tensor


def get_zarr_coords(z_store):
    """
    Extracts latitude and longitude from a Zarr group.
    Falls back to computing lat/lon from 3D Cartesian coordinates if lat/lon are NaNs.
    """
    lats, lons = None, None

    # 1. Try reading direct latitude / longitude arrays
    for lat_name in ["latitude", "lat", "clat"]:
        if lat_name in z_store:
            vals = z_store[lat_name][:]
            if not np.all(np.isnan(vals)):
                lats = vals
                break

    for lon_name in ["longitude", "lon", "clon"]:
        if lon_name in z_store:
            vals = z_store[lon_name][:]
            if not np.all(np.isnan(vals)):
                lons = vals
                break

    # 2. Fallback: Compute from Cartesian (x_cartesian, y_cartesian, z_cartesian)
    if lats is None or lons is None or np.all(np.isnan(lats)) or np.all(np.isnan(lons)):
        if "x_cartesian" in z_store and "y_cartesian" in z_store and "z_cartesian" in z_store:
            print("[INIT] 'latitude'/'longitude' contain NaNs. Computing lat/lon from (x, y, z) Cartesian coordinates...")
            x = z_store["x_cartesian"][:]
            y = z_store["y_cartesian"][:]
            z = z_store["z_cartesian"][:]

            # Calculate spherical coordinates in degrees
            r = np.sqrt(x**2 + y**2 + z**2)
            lats = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
            lons = np.degrees(np.arctan2(y, x))
            
            # Convert longitudes to [0, 360]
            lons = np.where(lons < 0, lons + 360, lons)

    if lats is None or lons is None or np.all(np.isnan(lats)) or np.all(np.isnan(lons)):
        raise ValueError(
            f"Could not extract valid coordinates from Zarr store! "
            f"Found keys: {list(z_store.keys())}"
        )

    # Convert units from radians to degrees if necessary
    if np.nanmax(lons) <= 2 * np.pi and np.nanmax(lats) <= np.pi:
        print("[INIT] Converting coordinate units from radians to degrees...")
        lons = np.degrees(lons)
        lats = np.degrees(lats)

    # Ensure positive longitudes [0, 360]
    lons = np.where(lons < 0, lons + 360, lons)

    return lats, lons


def save_bg_to_zarr_temp(x_bg_tensor, template_zarr, out_zarr_path):
    """
    Writes background prediction to a temporary Zarr directory for 3D-Var ingestion.
    Preserves valid spatial coordinates.
    """
    z_template = zarr.open(template_zarr, mode="r")
    lats, lons = get_zarr_coords(z_template)

    # Extract height if present, otherwise fall back to level or default heights
    if "height" in z_template:
        heights = z_template["height"][:]
    elif "level" in z_template:
        heights = z_template["level"][:]
    else:
        heights = np.arange(32, dtype=np.float32)

    bg_np = x_bg_tensor.cpu().numpy()[0]  # Shape: (5, 32, 2562)

    ds_dict = {
        "latitude": (["node"], lats),
        "longitude": (["node"], lons),
        "height": (["level"], heights),
    }

    for idx, var in enumerate(STATE_VARS):
        zarr_name = ZARR_VAR_MAP[var]
        # Shape expected by 3D-Var: (1, 32, 2562)
        ds_dict[zarr_name] = (["time", "level", "node"], bg_np[idx:idx+1])

    ds = xr.Dataset(ds_dict)
    ds.to_zarr(out_zarr_path, mode="w")


def main():
    parser = argparse.ArgumentParser(description="Run AI-DA Operational Cycling Loop")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYYMMDD_HH), e.g., 20230306_00")
    parser.add_argument("--num_cycles", type=int, default=4, help="Number of 6-hour cycles to run")
    parser.add_argument("--gnn_ckpt", type=str, default="../checkpoints/aida_gnn_v1.pt")
    parser.add_argument("--edges", type=str, default="../data/graph/edge_index_m4.pt")
    parser.add_argument("--zarr_init", type=str, default="../data/icosahedral_2023.zarr")
    parser.add_argument("--bmatrix", type=str, default="../bmatrix/bmatrix_from_gfs_analysis.nc")
    parser.add_argument("--obs_dir", type=str, default="../data/conv_2023")
    parser.add_argument("--work_dir", type=str, default="./cycling_output")

    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"[AIDA CYCLING] Initializing AI-DA Multi-Cycle Experiment")
    print(f"Device: {device} | Cycles: {args.num_cycles}")
    print(f"========================================================\n")

    # 1. Load Edge Topology and Instantiate GNN
    print(f"[GNN] Loading graph structure from '{args.edges}'...")
    edge_index = torch.load(args.edges, map_location=device).long()

    print(f"[GNN] Loading surrogate checkpoint from '{args.gnn_ckpt}'...")
    gnn_model = IcosahedralGraphGNN(
        edge_index=edge_index,
        in_channels=5*32,
        hidden_dim=128,
        out_channels=5*32
    ).to(device)
    gnn_model.load_state_dict(torch.load(args.gnn_ckpt, map_location=device))
    gnn_model.eval()

    # 2. Extract initial state at start_date
    current_dt = datetime.datetime.strptime(args.start_date, "%Y%m%d_%H")
    print(f"[INIT] Extracting initial analysis condition for {current_dt.strftime('%Y-%m-%d %H:00Z')}...")

    z_init = zarr.open(args.zarr_init, mode="r")
    x_init_list = [z_init[ZARR_VAR_MAP[v]][0] for v in STATE_VARS]
    x_current = torch.tensor(np.array(x_init_list), dtype=torch.float32).unsqueeze(0)  # (1, 5, 32, 2562)

    # Validate coordinate extraction right at startup
    get_zarr_coords(z_init)

    # 3. Main Cycling Loop
    for cycle in range(1, args.num_cycles + 1):
        next_dt = current_dt + datetime.timedelta(hours=6)
        date_str = next_dt.strftime("%Y%m%d")
        cycle_str = f"t{next_dt.strftime('%H')}z"

        print(f"\n--------------------------------------------------------")
        print(f">>> CYCLE {cycle}/{args.num_cycles}: {current_dt.strftime('%Y%m%d_%H')}Z -> {next_dt.strftime('%Y%m%d_%H')}Z")
        print(f"--------------------------------------------------------")

        # Step A: GNN 6-Hour Forecast Pass
        print(f"[STEP A] Running GNN 6-hour forecast step...")
        x_bg_tensor = run_gnn_forecast(gnn_model, x_current, device)

        # Save temporary background Zarr for 3D-Var
        temp_bg_zarr = os.path.join(args.work_dir, f"temp_bg_{date_str}_{cycle_str}.zarr")
        save_bg_to_zarr_temp(x_bg_tensor, args.zarr_init, temp_bg_zarr)

        # Step B: 3D-Var Data Assimilation
        obs_file = os.path.join(args.obs_dir, f"conv.{date_str}.{cycle_str}.nc")
        out_analysis = os.path.join(args.work_dir, f"aida_analysis_{date_str}_{cycle_str}.nc")

        print(f"[STEP B] Executing 3D-Var Assimilation with observation file:")
        print(f"         '{obs_file}'")

        cmd = [
            sys.executable, "scripts/run_aida_3dvar.py",
            "--zarr", temp_bg_zarr,
            "--bmatrix", args.bmatrix,
            "--edges", args.edges,
            "--conv", obs_file,
            "--output", out_analysis,
            "--maxiter", "30"
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"[ERROR] 3D-Var failed at cycle {cycle}. Exiting.")
            sys.exit(1)

        # Step C: Read Analysis Output to update initial condition for next cycle
        print(f"[STEP C] Updating current state from analysis '{out_analysis}'...")
        ds_an = xr.open_dataset(out_analysis)
        an_list = []
        for v in STATE_VARS:
            analysis_var_name = f"{v}_analysis"  # Reads 'p_analysis', 't_analysis', etc.
            an_data = ds_an[analysis_var_name].values  # Shape: (32, 2562)
            an_list.append(an_data)

        # Format as input tensor for GNN next cycle: shape (1, 5, 32, 2562)
        x_current = torch.tensor(np.array(an_list), dtype=torch.float32).unsqueeze(0)
        current_dt = next_dt

    print(f"\n========================================================")
    print(f"[COMPLETE] AI-DA Multi-Cycle Experiment Finished Successfully!")
    print(f"All analysis outputs saved in: '{args.work_dir}'")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()
