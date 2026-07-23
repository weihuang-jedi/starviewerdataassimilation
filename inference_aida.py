# inference_aida.py
import numpy as np
import torch
import xarray as xr
from model_gnn import IcosahedralAIDA_GNN
from train_aida import extract_edge_index_from_faces, load_nnja_obs_to_mesh
from scipy.spatial import KDTree


def run_inference_on_xb(
    checkpoint_path: str,
    xb_nc_file: str,
    obs_parquet_file: str,
    output_nc_file: str = "analysis_output.nc",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running AI-DA Inference on Device: {device}")

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    face_nodes = checkpoint["face_nodes"]
    edge_index = extract_edge_index_from_faces(face_nodes).to(device)

    # Initialize Model & Load Weights
    model = IcosahedralAIDA_GNN(num_nodes=2562, num_levels=32, in_vars=6).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[✓] Loaded model weights from checkpoint.")

    # 2. Read Target Input x_b NetCDF File
    ds_xb = xr.open_dataset(xb_nc_file)
    node_lats = ds_xb["latitude"].values
    node_lons = ds_xb["longitude"].values

    t_val = ds_xb["t_icosahedral"].values
    u_val = ds_xb["u_icosahedral"].values
    v_val = ds_xb["v_icosahedral"].values
    w_val = ds_xb["w_icosahedral"].values
    q_val = ds_xb["q_icosahedral"].values
    p_val = ds_xb["p_icosahedral"].values

    # Reshape input to (1, Node, Vars * Height)
    x_b_np = np.concatenate(
        [t_val, u_val, v_val, w_val, q_val, p_val], axis=0
    ).T
    x_b_tensor = torch.from_numpy(x_b_np).unsqueeze(0).to(device)

    # 3. Read NNJA-AI Observations and Map to Mesh Nodes
    kdtree = KDTree(np.column_stack([node_lats, node_lons]))
    obs_inno_np = load_nnja_obs_to_mesh(
        obs_parquet_file, node_lats, node_lons, kdtree, num_nodes=2562
    )
    obs_inno_tensor = torch.from_numpy(obs_inno_np).unsqueeze(0).to(device)

    # 4. Model Forward Pass
    with torch.no_grad():
        x_a_pred, dx_pred = model(x_b_tensor, obs_inno_tensor, edge_index)

    x_a_np = x_a_pred.squeeze(0).cpu().numpy().T  # Shape: (192, 2562)
    dx_np = dx_pred.squeeze(0).cpu().numpy().T  # Shape: (192, 2562)

    # Split output variables back into (height=32, node=2562)
    t_inc = dx_np[0:32, :]
    u_inc = dx_np[32:64, :]
    v_inc = dx_np[64:96, :]

    t_a = x_a_np[0:32, :]
    u_a = x_a_np[32:64, :]
    v_a = x_a_np[64:96, :]

    # 5. Export Results to NetCDF File
    ds_out = ds_xb.copy(deep=True)

    # Add Increments and Analysis to Dataset
    ds_out["t_increment"] = (("height", "node"), t_inc)
    ds_out["u_increment"] = (("height", "node"), u_inc)
    ds_out["v_increment"] = (("height", "node"), v_inc)

    ds_out["t_analysis"] = (("height", "node"), t_a)
    ds_out["u_analysis"] = (("height", "node"), u_a)
    ds_out["v_analysis"] = (("height", "node"), v_a)

    ds_out.to_netcdf(output_nc_file)
    print(f"[✓] Analysis $x_a$ and Increments $\delta x$ written to '{output_nc_file}'")


if __name__ == "__main__":
    run_inference_on_xb(
        checkpoint_path="checkpoints/icosahedral_aida_model.pt",
        xb_nc_file="data/icosahedral_grid/global_icosahedral_m4.20230101.t00z.1p00.f006.nc",
        obs_parquet_file="data/nnja_obs/nnja_obs_20230101_00z.parquet",
        output_nc_file="analysis_output_20230101.nc",
    )
