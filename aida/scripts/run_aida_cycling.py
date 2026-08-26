#!/usr/bin/env python3
"""
scripts/run_aida_cycling.py
---------------------------
AI Data Assimilation Analysis & Autoregressive Cycling Engine.
Infers Analysis A_0h from Background B_0h and Observations O_0h,
then steps the state forward autoregressively while preserving 
target_level, eta, and 3D terrain-following vertical coordinates.
"""

import argparse
import os
import sys
import re
import yaml
import numpy as np
import xarray as xr
import torch
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gnn import IcosahedralGNNSurrogate
from models.graph import generate_or_load_edge_index


def compute_solar_zenith_angle(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    year: int = 2026,
    month: int = 1,
    day: int = 1,
    hour_utc: float = 0.0
) -> np.ndarray:
    """Computes cosine of Solar Zenith Angle cos(SZA) across all mesh nodes."""
    rad = np.pi / 180.0
    from datetime import datetime
    dt = datetime(year, month, day)
    day_of_year = dt.timetuple().tm_yday

    declination = 23.45 * np.sin(rad * (360.0 / 365.0) * (day_of_year - 81)) * rad
    solar_time = hour_utc + (lons_deg / 15.0)
    hour_angle = (solar_time - 12.0) * 15.0 * rad

    lats_rad = lats_deg * rad
    cos_sza = np.sin(lats_rad) * np.sin(declination) + np.cos(lats_rad) * np.cos(declination) * np.cos(hour_angle)
    return np.maximum(0.0, cos_sza).astype(np.float32)


def load_state_from_file(file_path: str, var_names: list):
    """Helper to extract dynamic 7-variable log-state tensor, 3D terrain heights, static topography, and coordinates."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Input state file not found: '{file_path}'")

    if file_path.endswith('.zarr'):
        ds = xr.open_zarr(file_path)
    else:
        ds = xr.open_dataset(file_path)

    state_vars = []
    for v in var_names:
        if v in ds:
            val = ds[v].values
        elif f"{v}_icosahedral" in ds:
            val = ds[f"{v}_icosahedral"].values
        else:
            raise KeyError(f"[ERROR] Required state variable '{v}' missing in '{file_path}'")

        if val.ndim == 3:
            val = val[0]
        state_vars.append(val)

    state_np = np.stack(state_vars, axis=0).astype(np.float32)

    lats = ds['latitude'].values if 'latitude' in ds else (ds['lat'].values if 'lat' in ds else np.linspace(-90, 90, state_np.shape[2]))
    lons = ds['longitude'].values if 'longitude' in ds else (ds['lon'].values if 'lon' in ds else np.linspace(-180, 180, state_np.shape[2]))

    if lats.ndim > 1:
        lats = lats[0]
    if lons.ndim > 1:
        lons = lons[0]

    lons = np.where(lons > 180.0, lons - 360.0, lons)

    # 3D Height Profile Extract / M6 Formula Compute: h = Hmax - eta*(Hmax - Hterrain)
    if 'h_icosahedral' in ds:
        h_3d_np = ds['h_icosahedral'].values
    elif 'eta' in ds:
        eta_vals = ds['eta'].values
        Hmax = 20000.0
        h_terrain_val = ds['h_terrain_icosahedral'].values if 'h_terrain_icosahedral' in ds else np.zeros((state_np.shape[2],))
        if h_terrain_val.ndim > 1:
            h_terrain_val = h_terrain_val[0]
        h_3d_np = Hmax - eta_vals[:, np.newaxis] * (Hmax - h_terrain_val[np.newaxis, :])
    else:
        num_levels = state_np.shape[1]
        num_nodes = state_np.shape[2]
        baseline_h = np.linspace(2, 20000, num_levels, dtype=np.float32)
        h_3d_np = np.repeat(baseline_h[:, np.newaxis], num_nodes, axis=1)

    if h_3d_np.ndim == 3:
        h_3d_np = h_3d_np[0]

    # Surface Features
    if 'h_terrain_icosahedral' in ds:
        h_terrain = ds['h_terrain_icosahedral'].values
    elif 'elevation' in ds:
        h_terrain = ds['elevation'].values
    else:
        h_terrain = np.zeros((state_np.shape[2],), dtype=np.float32)

    if 'land_sea_mask' in ds:
        ls_mask = ds['land_sea_mask'].values
    else:
        ls_mask = np.zeros((state_np.shape[2],), dtype=np.float32)

    if h_terrain.ndim > 1:
        h_terrain = h_terrain[0]
    if ls_mask.ndim > 1:
        ls_mask = ls_mask[0]

    # Surface Roughness z0
    z0_map = np.where(ls_mask > 0.5, 0.1, 0.0002).astype(np.float32)
    ln_z0_norm = (np.log(z0_map + 1e-5) / 5.0).astype(np.float32)

    # Base static topo: 3 channels [Elevation, LSM, z0]
    static_topo_base = np.stack([
        h_terrain.astype(np.float32) / 10000.0,
        ls_mask.astype(np.float32),
        ln_z0_norm
    ], axis=0)

    return state_np, h_3d_np.astype(np.float32), static_topo_base, lats.astype(np.float32), lons.astype(np.float32), ds


def parse_date_tag(filename: str) -> tuple[str, int, int, int, int]:
    match = re.search(r'(\d{4})(\d{2})(\d{2})\.t(\d{2})z', os.path.basename(filename))
    if match:
        year, month, day, hour = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        date_tag = match.group(0).replace('.nc', '')
        return date_tag, year, month, day, hour
    return "forecast", 2026, 1, 1, 0


def export_lead_time_netcdf(
    output_path: str,
    state_arr: np.ndarray,
    h_3d_np: np.ndarray,
    static_topo_np: np.ndarray,
    ds_ref: xr.Dataset,
    var_names: list,
    lead_time_hours: int,
    num_levels: int,
    num_nodes: int
):
    """Saves forecast state as CF/UGRID NetCDF while copying target_level and eta from ds_ref."""
    data_vars_out = {}

    for idx, var in enumerate(var_names):
        out_var_name = var if var.endswith("_icosahedral") else f"{var}_icosahedral"
        var_attrs = {"long_name": f"Forecasted {var}", "mesh": "icosahedral_mesh"}
        if out_var_name in ds_ref:
            var_attrs.update(ds_ref[out_var_name].attrs)

        data_vars_out[out_var_name] = (
            ["level", "node"],
            state_arr[idx, :, :].astype(np.float32),
            var_attrs
        )

    h_attrs = {"units": "meters", "long_name": "3D Terrain-Following Geometric Height Above Sea Level", "mesh": "icosahedral_mesh"}
    if "h_icosahedral" in ds_ref:
        h_attrs.update(ds_ref["h_icosahedral"].attrs)

    data_vars_out["h_icosahedral"] = (["level", "node"], h_3d_np, h_attrs)
    data_vars_out["h_terrain_icosahedral"] = (["node"], static_topo_np[0] * 10000.0, {"units": "meters", "mesh": "icosahedral_mesh"})

    # Preserve target_level and eta from reference file
    if "eta" in ds_ref:
        data_vars_out["eta"] = (["level"], ds_ref["eta"].values, ds_ref["eta"].attrs)
    if "target_level" in ds_ref:
        data_vars_out["target_level"] = (["level"], ds_ref["target_level"].values, ds_ref["target_level"].attrs)

    static_vars = ["longitude", "latitude", "face_nodes", "x_cartesian", "y_cartesian", "z_cartesian", "land_sea_mask", "elevation", "h_terrain", "icosahedral_mesh"]
    for static_var in static_vars:
        if static_var in ds_ref:
            data_vars_out[static_var] = ds_ref[static_var]

    coords_out = {
        "level": ds_ref["level"].values if "level" in ds_ref else np.arange(1, num_levels + 1, dtype=np.int32),
        "node": ds_ref["node"].values if "node" in ds_ref else np.arange(num_nodes, dtype=np.int32)
    }

    if "face" in ds_ref.dims:
        coords_out["face"] = ds_ref["face"].values
    if "three" in ds_ref.dims:
        coords_out["three"] = ds_ref["three"].values

    ds_out = xr.Dataset(
        data_vars=data_vars_out,
        coords=coords_out,
        attrs={
            "title": getattr(ds_ref, "title", "AIDA GNN 4D Terrain Weather Forecast"),
            "conventions": "CF-1.8 UGRID-1.0",
            "forecast_lead_time_hours": lead_time_hours
        }
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ds_out.to_netcdf(output_path, format="NETCDF4")
    ds_out.close()
    print(f"  ├─ Saved lead time f{lead_time_hours:03d}h -> '{output_path}' (copied target_level & eta)", flush=True)


def run_aida_cycling(
    ckpt_path: str,
    background_file: str,
    edge_index_path: str,
    forecast_steps: int = 4,
    output_pattern: str = "output/aida.{date_tag}.f{lead:03d}.nc"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CYCLING] Operating on compute device: {device}", flush=True)

    print(f"[CYCLING] Loading checkpoint: '{ckpt_path}'", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", {})

    model_cfg = cfg.get("model", {})
    hidden_dim = model_cfg.get("hidden_dim", 128)
    num_levels = cfg.get("mesh", {}).get("num_levels", 32)
    num_layers = model_cfg.get("num_layers", 4)

    total_in_vars = 25
    out_vars = 7

    model = IcosahedralGNNSurrogate(
        in_vars=total_in_vars,
        out_vars=out_vars,
        hidden_dim=hidden_dim,
        num_levels=num_levels,
        num_layers=num_layers
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    var_names = [
        'ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
        'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral'
    ]

    print(f"[CYCLING] Reading Background State B_0h: '{background_file}'", flush=True)
    b_0h_np, h_3d_np, static_topo_base_np, lats_deg, lons_deg, ds_ref = load_state_from_file(background_file, var_names)

    date_tag, base_year, base_month, base_day, base_hour_utc = parse_date_tag(background_file)
    num_nodes = b_0h_np.shape[2]
    edge_index = generate_or_load_edge_index(num_nodes=num_nodes, edge_file=edge_index_path).to(device)

    b_0h_tensor = torch.from_numpy(b_0h_np).unsqueeze(0).to(device)                 # [1, 7, 32, Nodes]
    static_topo_base = torch.from_numpy(static_topo_base_np).unsqueeze(0).to(device) # [1, 3, Nodes]

    print(f"\n" + "=" * 80)
    print(f" STARTING AI-DA ANALYSIS & FORECAST CYCLING ROLLOUT")
    print(f" Base Time Tag: {date_tag} (Year:{base_year}, Month:{base_month}, Day:{base_day}, Hour:{base_hour_utc:02d}z)")
    print("=" * 80, flush=True)

    with torch.no_grad():
        cos_sza_np = compute_solar_zenith_angle(
            lats_deg=lats_deg,
            lons_deg=lons_deg,
            year=base_year,
            month=base_month,
            day=base_day,
            hour_utc=base_hour_utc
        )

        cos_sza_tensor = torch.from_numpy(cos_sza_np).unsqueeze(0).unsqueeze(0).to(device)
        static_topo_4ch = torch.cat([static_topo_base, cos_sza_tensor], dim=1)            # [1, 4, Nodes]
        static_topo_exp = static_topo_4ch.unsqueeze(2).expand(-1, -1, num_levels, -1)     # [1, 4, 32, Nodes]

        # Formulate 25-channel Input: [B_0h (7), Conv_Val (7), Conv_Mask (7), Static_Topo (4)]
        obs_val_placeholder = torch.zeros_like(b_0h_tensor)
        obs_mask_placeholder = torch.zeros_like(b_0h_tensor)

        x_input_0h = torch.cat([b_0h_tensor, obs_val_placeholder, obs_mask_placeholder, static_topo_exp], dim=1)

        # Infer Analysis at 0h (A_0h)
        a_0h_tensor = model(x_input_0h, edge_index)

        # Save f000 Analysis State
        f000_path = output_pattern.format(date_tag=date_tag, lead=0)
        export_lead_time_netcdf(
            output_path=f000_path,
            state_arr=a_0h_tensor.cpu().numpy().squeeze(0),
            h_3d_np=h_3d_np,
            static_topo_np=static_topo_base_np,
            ds_ref=ds_ref,
            var_names=var_names,
            lead_time_hours=0,
            num_levels=num_levels,
            num_nodes=num_nodes
        )

        # Autoregressive Cycling Steps
        state_curr = a_0h_tensor
        for step in range(1, forecast_steps + 1):
            lead_hours = step * 6
            current_hour_utc = (base_hour_utc + lead_hours) % 24

            cos_sza_step = compute_solar_zenith_angle(
                lats_deg=lats_deg,
                lons_deg=lons_deg,
                year=base_year,
                month=base_month,
                day=base_day,
                hour_utc=current_hour_utc
            )

            cos_sza_step_tensor = torch.from_numpy(cos_sza_step).unsqueeze(0).unsqueeze(0).to(device)
            static_topo_step = torch.cat([static_topo_base, cos_sza_step_tensor], dim=1)
            static_topo_step_exp = static_topo_step.unsqueeze(2).expand(-1, -1, num_levels, -1)

            x_input_step = torch.cat([state_curr, obs_val_placeholder, obs_mask_placeholder, static_topo_step_exp], dim=1)

            out_model = model(x_input_step, edge_index)

            delta_max = torch.tensor([0.035, 20.0, 20.0, 2.0, 0.005, 0.1, 0.02], device=device).view(1, 7, 1, 1)
            out_model = torch.clamp(out_model, min=-delta_max, max=delta_max)

            state_next = state_curr + out_model

            # Physical Guards
            state_next[:, 0, :, :] = torch.clamp(state_next[:, 0, :, :], min=5.19295, max=5.79909)
            state_next[:, 1, :, :] = torch.clamp(state_next[:, 1, :, :], min=-90.0, max=90.0)
            state_next[:, 2, :, :] = torch.clamp(state_next[:, 2, :, :], min=-90.0, max=90.0)
            state_next[:, 4, :, :] = torch.clamp(state_next[:, 4, :, :], min=0.0, max=0.035)
            state_next[:, 6, :, :] = torch.clamp(state_next[:, 6, :, :], min=4.60517, max=11.58988)

            next_np = state_next.cpu().numpy().squeeze(0)

            step_path = output_pattern.format(date_tag=date_tag, lead=lead_hours)
            export_lead_time_netcdf(
                output_path=step_path,
                state_arr=next_np,
                h_3d_np=h_3d_np,
                static_topo_np=static_topo_base_np,
                ds_ref=ds_ref,
                var_names=var_names,
                lead_time_hours=lead_hours,
                num_levels=num_levels,
                num_nodes=num_nodes
            )

            state_curr = state_next

    ds_ref.close()
    print(f"\n[SUCCESS] AI-DA Analysis & Forecast Cycling rollout complete! Exported {forecast_steps + 1} NetCDF files.\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="AIDA 4D Terrain-Following AI-DA Cycling Engine")
    parser.add_argument("-k", "--checkpoint", default="checkpoints/aida_gnn_surrogate_logstate.pt", help="Path to checkpoint")
    parser.add_argument("-m", "--background", required=True, help="Path to B_0h background state file")
    parser.add_argument("-e", "--edges", default="data/graph/icosahedral_edge_index_m6.pt", help="Path to graph edge index")
    parser.add_argument("-s", "--steps", type=int, default=4, help="Number of 6h forecast steps (default: 4 = 24h)")
    parser.add_argument("-o", "--output_pattern", default="output/aida.{date_tag}.f{lead:03d}.nc", help="Output path pattern")

    args = parser.parse_args()

    run_aida_cycling(
        ckpt_path=args.checkpoint,
        background_file=args.background,
        edge_index_path=args.edges,
        forecast_steps=args.steps,
        output_pattern=args.output_pattern
    )


if __name__ == "__main__":
    main()
