#!/usr/bin/env python3
"""
scripts/train_aida_surrogate.py
-------------------------------
DA-Aware Offline Training Loop for the Icosahedral Mesh GNN Surrogate Model.
Predicts x_{t+6h} from x_t using graph message passing and uses B-matrix 
error variances to weight the training loss.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    "p": "p_var",
    "t": "t_var",
    "u": "u_var",
    "v": "v_var",
    "q": "q_var",
}


# ==============================================================================
# 1. GRAPH NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class IcosahedralGraphGNN(nn.Module):
    """
    Graph Neural Network Surrogate Model for Icosahedral Mesh.
    Predicts residual update dx: x_{t+6h} = x_t + dx(x_t)
    """
    def __init__(self, edge_index, in_channels=5*32, hidden_dim=128, out_channels=5*32):
        super().__init__()
        # Store edge index (source and destination node indices)
        self.register_buffer("edge_index", edge_index)
        self.src = edge_index[0]
        self.dst = edge_index[1]

        # Node feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Message Passing Layer (Graph Edge Interactions)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Node feature decoder (Predicts state increment dx)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x):
        # Input shape: (batch_size, 5_vars, 32_levels, 2562_nodes)
        batch_size, n_vars, n_lvls, n_nodes = x.shape

        # Reshape to (batch_size, 2562_nodes, 160_features)
        x_flat = x.view(batch_size, n_vars * n_lvls, n_nodes).permute(0, 2, 1)

        # 1. Encode node features
        h = self.encoder(x_flat)  # Shape: (batch_size, 2562, hidden_dim)

        # 2. Graph Message Passing across Mesh Edges
        h_src = h[:, self.src, :]  # (batch_size, 15360_edges, hidden_dim)
        h_dst = h[:, self.dst, :]  # (batch_size, 15360_edges, hidden_dim)

        msg_in = torch.cat([h_src, h_dst], dim=-1)
        messages = self.msg_mlp(msg_in)

        # Aggregate incoming messages at destination nodes along dim 1
        # self.dst is a 1D vector of shape [15360]
        aggr_msg = torch.zeros_like(h)
        aggr_msg.index_add_(1, self.dst, messages)

        h_out = h + aggr_msg

        # 3. Decode state increment dx
        dx_flat = self.decoder(h_out).permute(0, 2, 1).view(batch_size, n_vars, n_lvls, n_nodes)

        # Residual prediction formulation: x_{t+6h} = x_t + dx
        return x + dx_flat


# ==============================================================================
# 2. B-MATRIX LOSS WEIGHTING LOGIC
# ==============================================================================
def load_and_map_bmatrix_weights(bmatrix_path, mesh_lats, mesh_lons):
    """
    Interpolates the (32, 181, 360) B-matrix error variances onto the icosahedral grid,
    and returns normalized loss weights: W = 1.0 / (\sigma_b^2).
    """
    print(f"[B-MATRIX] Loading and mapping error variances from '{bmatrix_path}'...")
    ds_b = xr.open_dataset(bmatrix_path)

    b_lons = ds_b["longitude"].values
    b_lons = np.where(b_lons > 180, b_lons - 360, b_lons)
    ds_b = ds_b.assign_coords(longitude=b_lons).sortby("longitude")

    loss_weights_dict = {}

    for var in STATE_VARS:
        b_var_name = VAR_NAME_MAP_BMATRIX[var]
        var_data = ds_b[b_var_name]

        # Interpolate variances to (32, 2562)
        interpolated = var_data.interp(
            latitude=xr.DataArray(mesh_lats, dims="node"),
            longitude=xr.DataArray(mesh_lons, dims="node"),
            method="linear"
        ).values

        interpolated = np.nan_to_num(interpolated, nan=1.0)
        variance = np.maximum(interpolated, 1e-8)

        # Weight inversely proportional to background variance
        inv_var = 1.0 / variance
        norm_weight = inv_var / np.mean(inv_var)
        loss_weights_dict[var] = torch.tensor(norm_weight, dtype=torch.float32)

    weights_tensor = torch.stack([loss_weights_dict[v] for v in STATE_VARS], dim=0)
    print(f"  -> Mapped B-Matrix loss weights to mesh: shape {weights_tensor.shape}")
    return weights_tensor


# ==============================================================================
# 3. ZARR DATASET LOADER
# ==============================================================================
class IcosahedralZarrDataset(Dataset):
    """Loads consecutive (x_t, x_{t+6h}) pairs from background Zarr file."""
    def __init__(self, zarr_path, lead_steps=1):
        self.z_root = zarr.open(zarr_path, mode="r")
        self.lead_steps = lead_steps
        self.n_times = self.z_root["p_icosahedral"].shape[0] - lead_steps

    def __len__(self):
        return self.n_times

    def __getitem__(self, idx):
        x_t = []
        x_target = []
        for var in STATE_VARS:
            zarr_name = ZARR_VAR_MAP[var]
            x_t.append(self.z_root[zarr_name][idx])
            x_target.append(self.z_root[zarr_name][idx + self.lead_steps])

        return (
            torch.tensor(np.array(x_t), dtype=torch.float32),
            torch.tensor(np.array(x_target), dtype=torch.float32)
        )


# ==============================================================================
# 4. MAIN TRAINING LOOP
# ==============================================================================
def train_model():
    parser = argparse.ArgumentParser(description="Train DA-Aware AI-DA GNN Forecast Model")
    parser.add_argument("--zarr", type=str, default="../data/icosahedral_2023.zarr")
    parser.add_argument("--bmatrix", type=str, default="../bmatrix/bmatrix_from_gfs_analysis.nc")
    parser.add_argument("--edges", type=str, default="../data/graph/edge_index_m4.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_ckpt", type=str, default="../checkpoints/aida_gnn_v1.pt")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SYSTEM] Running on Compute Device: {device}")

    # 1. Read mesh coordinates
    z_root = zarr.open(args.zarr, mode="r")
    mesh_lats = np.array(z_root["latitude"][:], dtype=np.float32)
    mesh_lons = np.array(z_root["longitude"][:], dtype=np.float32)
    mesh_lons = np.where(mesh_lons > 180, mesh_lons - 360, mesh_lons)

    # 2. Load B-matrix loss weights
    loss_weights = load_and_map_bmatrix_weights(args.bmatrix, mesh_lats, mesh_lons).to(device)

    # 3. Load Graph Topology Edges
    print(f"[GRAPH] Loading edge topology from '{args.edges}'...")
    edge_index = torch.load(args.edges, map_location=device).long()

    # 4. Instantiate GNN Model, Optimizer, and Loss
    print(f"[MODEL] Initializing IcosahedralGraphGNN Architecture...")
    model = IcosahedralGraphGNN(
        edge_index=edge_index,
        in_channels=5*32,
        hidden_dim=128,
        out_channels=5*32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 5. Load Dataset
    dataset = IcosahedralZarrDataset(args.zarr)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print(f"\n[TRAINING] Starting Training Cycle...")
    print(f"  -> Total Samples: {len(dataset)} | Batch Size: {args.batch_size} | Total Batches: {len(dataloader)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x_in, y_true in dataloader:
            x_in, y_true = x_in.to(device), y_true.to(device)

            optimizer.zero_grad()

            # Forward Pass through GNN
            y_pred = model(x_in)  # Shape: (batch, 5, 32, 2562)

            # Compute B-matrix weighted MSE Loss
            squared_diff = (y_pred - y_true) ** 2
            weighted_diff = squared_diff * loss_weights.unsqueeze(0)  # Broadcast batch dim
            loss = torch.mean(weighted_diff)

            # Backward Pass and Weight Update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | B-Matrix Weighted Loss: {avg_loss:.6f}")

    # Save trained checkpoint
    torch.save(model.state_dict(), args.out_ckpt)
    print(f"\n[COMPLETE] Model checkpoint successfully saved to: '{args.out_ckpt}'")


if __name__ == "__main__":
    train_model()
