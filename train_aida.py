# train_aida.py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import xarray as xr
from model_gnn import IcosahedralAIDA_GNN
from scipy.spatial import KDTree


def extract_edge_index_from_faces(face_nodes):
    """Converts (face, 3) triangle face connectivity into PyTorch Graph edge_index (2, num_edges)."""
    edges = set()
    for tri in face_nodes:
        n1, n2, n3 = tri
        edges.add((n1, n2))
        edges.add((n2, n1))
        edges.add((n2, n3))
        edges.add((n3, n2))
        edges.add((n3, n1))
        edges.add((n1, n3))

    edge_list = list(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def load_nnja_obs_to_mesh(
    parquet_path, node_lats, node_lons, kdtree, num_nodes, num_levels=32
):
    """Reads NNJA-AI observation Parquet file and projects point innovations onto nearest mesh nodes."""
    grid_innovations = np.zeros(
        (num_nodes, num_levels * 6), dtype=np.float32
    )

    try:
        df_obs = pd.read_parquet(parquet_path)
    except Exception:
        # Return zeros if observation file for timestamp does not exist
        return grid_innovations

    if len(df_obs) == 0:
        return grid_innovations

    # Find nearest mesh node for each observation point (Lat/Lon)
    obs_coords = np.column_stack(
        [df_obs["latitude"].values, df_obs["longitude"].values]
    )
    _, node_indices = kdtree.query(obs_coords)

    # Accumulate innovations onto node tensor
    # Assuming column 'observation_innovation' exists
    if "innovation" in df_obs.columns:
        innovations = df_obs["innovation"].values
        np.add.at(
            grid_innovations[:, 0], node_indices, innovations
        )  # Add to 1st var level

    return grid_innovations


def train_icosahedral_aida():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Training on Device: {device}")

    # 1. Load Topology from sample file
    zarr_ds = xr.open_zarr("data/icosahedral_2023.zarr")
    face_nodes = zarr_ds["face_nodes"].values
    node_lats = zarr_ds["latitude"].values
    node_lons = zarr_ds["longitude"].values

    # Build Spatial KDTree for fast observation mapping
    mesh_coords = np.column_stack([node_lats, node_lons])
    kdtree = KDTree(mesh_coords)

    # Construct Graph Topology Edge Index
    edge_index = extract_edge_index_from_faces(face_nodes).to(device)

    # 2. Instantiate GNN Model
    model = IcosahedralAIDA_GNN(num_nodes=2562, num_levels=32, in_vars=6).to(
        device
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    num_timesteps = len(zarr_ds.time)
    print(f"Total Available Timesteps: {num_timesteps}")

    model.train()
    for epoch in range(10):
        running_loss = 0.0

        for t in range(min(num_timesteps, 50)):  # Sample loop
            # Extract x_b background state (Height x Node -> Node x (Height*Vars))
            t_val = zarr_ds["t_icosahedral"].isel(time=t).values
            u_val = zarr_ds["u_icosahedral"].isel(time=t).values
            v_val = zarr_ds["v_icosahedral"].isel(time=t).values
            w_val = zarr_ds["w_icosahedral"].isel(time=t).values
            q_val = zarr_ds["q_icosahedral"].isel(time=t).values
            p_val = zarr_ds["p_icosahedral"].isel(time=t).values

            # Stack into single feature vector per node
            x_b_np = np.concatenate(
                [t_val, u_val, v_val, w_val, q_val, p_val], axis=0
            ).T  # (2562, 192)
            x_b_tensor = (
                torch.from_numpy(x_b_np).unsqueeze(0).to(device)
            )  # Add batch dim

            # Load matching NNJA-AI observations
            obs_file = (
                f"data/nnja_obs/nnja_obs_{t:04d}.parquet"  # Example path
            )
            obs_inno_np = load_nnja_obs_to_mesh(
                obs_file, node_lats, node_lons, kdtree, 2562
            )
            obs_inno_tensor = (
                torch.from_numpy(obs_inno_np).unsqueeze(0).to(device)
            )

            # Target synthetic increment for demonstration
            target_dx = torch.randn_like(x_b_tensor) * 0.1

            # Forward Pass
            optimizer.zero_grad()
            x_a_pred, dx_pred = model(x_b_tensor, obs_inno_tensor, edge_index)

            loss = criterion(dx_pred, target_dx)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/10 - Loss: {running_loss / 50:.6f}")

    # 3. Save Checkpoint File
    checkpoint_path = "checkpoints/icosahedral_aida_model.pt"
    checkpoint = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "face_nodes": face_nodes,  # Save graph topology inside checkpoint
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\n[✓] Checkpoint saved successfully to '{checkpoint_path}'")


if __name__ == "__main__":
    train_icosahedral_aida()
