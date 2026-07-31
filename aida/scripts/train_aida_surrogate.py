#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
Trains the AIDA GNN surrogate model on scale-invariant log-state Zarr stores
[ln_T, u, v, w, q, ln_rho, ln_p] using Non-Hydrostatic Icosahedral Loss constraints.
"""

import os
import argparse
import numpy as np
import zarr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing


# Log-state variable configuration
LOG_STATE_VARS = [
    'ln_t_icosahedral',
    'u_icosahedral',
    'v_icosahedral',
    'w_icosahedral',
    'q_icosahedral',
    'ln_rho_icosahedral',
    'ln_p_icosahedral'
]

# ==========================================
# 1. GRAPH CONVOLUTION LAYER
# ==========================================
class IcosahedralGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(IcosahedralGraphConv, self).__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=self.lin(x))

    def message(self, x_j):
        return x_j


# ==========================================
# 2. LOG-STATE ZARR DATASET LOADER
# ==========================================
class LogStateZarrDataset(Dataset):
    def __init__(self, zarr_path, var_names=None, sequence_len=1):
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr store not found at: {zarr_path}")

        self.root = zarr.open(zarr_path, mode='r')
        self.var_names = var_names if var_names else LOG_STATE_VARS
        self.sequence_len = sequence_len

        # Probe dimensions using first variable
        first_var = self.root[self.var_names[0]]
        self.total_timesteps, self.num_levels, self.num_nodes = first_var.shape

        print("========================================================")
        print("[AIDA DATASET] Initialized Log-State Zarr Loader")
        print(f"Store Path  : {zarr_path}")
        print(f"Variables   : {self.var_names}")
        print(f"Timesteps   : {self.total_timesteps}")
        print(f"Mesh Layout : {self.num_levels} height levels x {self.num_nodes} nodes")
        print("========================================================\n")

        # Calculate Z-score stats per log variable
        self.stats = {}
        for var in self.var_names:
            arr = self.root[var]
            sample_data = np.asarray(arr[:min(100, self.total_timesteps)])
            sample_data = sample_data[np.isfinite(sample_data)]
            
            mean = float(np.mean(sample_data)) if sample_data.size > 0 else 0.0
            std  = float(np.std(sample_data)) if sample_data.size > 0 else 1.0

            if std < 1e-6 or np.isnan(std):
                std = 1.0
            if np.isnan(mean):
                mean = 0.0

            self.stats[var] = {'mean': mean, 'std': std}
            print(f"  -> [{var}] Mean: {mean:.4f} | Std: {std:.4f}")

    def __len__(self):
        return self.total_timesteps - self.sequence_len

    def __getitem__(self, idx):
        x_vars, y_vars = [], []

        for var in self.var_names:
            arr = self.root[var]
            mean = self.stats[var]['mean']
            std  = self.stats[var]['std']

            raw_x = np.nan_to_num(np.asarray(arr[idx], dtype=np.float32), nan=mean, posinf=mean, neginf=mean)
            raw_y = np.nan_to_num(np.asarray(arr[idx + self.sequence_len], dtype=np.float32), nan=mean, posinf=mean, neginf=mean)

            data_x = (raw_x - mean) / std
            data_y = (raw_y - mean) / std

            x_vars.append(data_x)
            y_vars.append(data_y)

        x_tensor = torch.tensor(np.stack(x_vars, axis=0), dtype=torch.float32)
        y_tensor = torch.tensor(np.stack(y_vars, axis=0), dtype=torch.float32)

        return x_tensor, y_tensor


# ==========================================
# 3. NON-HYDROSTATIC PHYSICAL LOSS
# ==========================================
class NonHydrostaticIcosahedralLoss(nn.Module):
    def __init__(
        self,
        means: dict,
        stds: dict,
        weight_mse: float = 1.0,
        weight_state_eq: float = 0.5,
        weight_mass: float = 0.2,
        weight_geo: float = 0.5,
        R_d: float = 287.058
    ):
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_state_eq = weight_state_eq
        self.weight_mass = weight_mass
        self.weight_geo = weight_geo
        self.R_d = R_d
        self.ln_R_d = np.log(R_d)

        # Register dataset mean and std for online un-normalization
        # Order expected in state tensor: [0: ln_t, 1: u, 2: v, 3: w, 4: q, 5: ln_rho, 6: ln_p]
        self.register_buffer("mu_ln_t", torch.tensor(means['ln_t_icosahedral'], dtype=torch.float32))
        self.register_buffer("std_ln_t", torch.tensor(stds['ln_t_icosahedral'], dtype=torch.float32))

        self.register_buffer("mu_ln_rho", torch.tensor(means['ln_rho_icosahedral'], dtype=torch.float32))
        self.register_buffer("std_ln_rho", torch.tensor(stds['ln_rho_icosahedral'], dtype=torch.float32))

        self.register_buffer("mu_ln_p", torch.tensor(means['ln_p_icosahedral'], dtype=torch.float32))
        self.register_buffer("std_ln_p", torch.tensor(stds['ln_p_icosahedral'], dtype=torch.float32))

        self.mse_fn = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, edge_index: torch.Tensor) -> dict:
        # 1. Base MSE Loss on Normalized State
        loss_mse = self.mse_fn(pred, target)

        # 2. Un-normalize log-thermodynamic variables to physical log-space
        # Shape: [Batch, Levels, Nodes]
        ln_T_phys   = pred[:, 0, :, :] * self.std_ln_t + self.mu_ln_t
        ln_rho_phys = pred[:, 5, :, :] * self.std_ln_rho + self.mu_ln_rho
        ln_p_phys   = pred[:, 6, :, :] * self.std_ln_p + self.mu_ln_p

        # 3. Un-normalized Ideal Gas State Equation Residual
        # ln(p) = ln(rho) + ln(R_d) + ln(T) -> Residual = ln_p - (ln_rho + ln_R_d + ln_T)
        state_eq_residual = ln_p_phys - (ln_rho_phys + self.ln_R_d + ln_T_phys)
        loss_state_eq = torch.mean(torch.abs(state_eq_residual))

        # 4. Global Mass Conservation Penalty (Density Integral Stability)
        loss_mass = torch.mean(torch.abs(torch.mean(pred[:, 5, :, :], dim=-1)))

        # 5. Geostrophic Balance Loss (Spatial Gradients over Graph Edges)
        u_pred = pred[:, 1, :, :]
        v_pred = pred[:, 2, :, :]
        src, dst = edge_index[0], edge_index[1]

        # Spatial gradient proxy on graph topology
        dp_edge = torch.abs(ln_p_phys[:, :, src] - ln_p_phys[:, :, dst])
        wind_mag = torch.sqrt(u_pred[:, :, src]**2 + v_pred[:, :, src]**2 + 1e-6)
        loss_geo = torch.mean(torch.abs(dp_edge - wind_mag * 0.1))

        # Total Weighted Non-Hydrostatic Loss
        total_loss = (
            self.weight_mse * loss_mse +
            self.weight_state_eq * loss_state_eq +
            self.weight_mass * loss_mass +
            self.weight_geo * loss_geo
        )

        return {
            "loss": total_loss,
            "mse": loss_mse.item(),
            "state_eq": loss_state_eq.item(),
            "mass": loss_mass.item(),
            "geo": loss_geo.item()
        }

        # return total_loss, loss_mse, loss_state_eq, loss_mass, loss_geo_mid


# ==========================================
# 4. ICOSAHEDRAL GNN SURROGATE
# ==========================================
class IcosahedralGNNSurrogate(nn.Module):
    def __init__(self, edge_index, in_vars=7, levels=32, hidden_dim=128):
        super(IcosahedralGNNSurrogate, self).__init__()
        self.register_buffer('edge_index', edge_index)
        in_features = in_vars * levels

        self.conv1 = IcosahedralGraphConv(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()

        self.conv2 = IcosahedralGraphConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()

        self.conv3 = IcosahedralGraphConv(hidden_dim, in_features)

    def forward(self, x):
        batch_size, num_vars, levels, num_nodes = x.shape
        x_flat = x.permute(0, 3, 1, 2).reshape(batch_size * num_nodes, num_vars * levels)

        if batch_size > 1:
            edge_list = [self.edge_index + (b * num_nodes) for b in range(batch_size)]
            batched_edges = torch.cat(edge_list, dim=1)
        else:
            batched_edges = self.edge_index

        h = self.act1(self.norm1(self.conv1(x_flat, batched_edges)))
        h = self.act2(self.norm2(self.conv2(h, batched_edges))) + (0.1 * h)
        out = self.conv3(h, batched_edges)

        return out.view(batch_size, num_nodes, num_vars, levels).permute(0, 2, 3, 1)


def generate_or_load_edge_index(num_nodes=2562, edge_file=None):
    if edge_file and os.path.exists(edge_file):
        print(f"[AIDA GRAPH] Loading edge topology from: {edge_file}")
        return torch.load(edge_file, weights_only=False)
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


# ==========================================
# 5. TRAINING EXECUTOR
# ==========================================
def run_training(zarr_path, edge_file, checkpoint_path, epochs, batch_size, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[AIDA TRAINING] Execution Device: {device}")

    dataset = LogStateZarrDataset(zarr_path=zarr_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    edge_index = generate_or_load_edge_index(num_nodes=dataset.num_nodes, edge_file=edge_file).to(device)

    model = IcosahedralGNNSurrogate(
        edge_index=edge_index,
        in_vars=len(dataset.var_names),
        levels=dataset.num_levels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = NonHydrostaticIcosahedralLoss(num_nodes=dataset.num_nodes).to(device)

    print(f"[AIDA TRAINING] Starting execution for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, mse_acc, state_acc, mass_acc, geo_acc = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)

            total_loss, l_mse, l_state, l_mass, l_geo = criterion(y_pred, y_batch)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tot_loss += total_loss.item()
            mse_acc += l_mse.item()
            state_acc += l_state.item()
            mass_acc += l_mass.item()
            geo_acc += l_geo.item()

        num_batches = len(dataloader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Total: {tot_loss/num_batches:.5f} | "
              f"MSE: {mse_acc/num_batches:.5f} | "
              f"StateEq: {state_acc/num_batches:.5f} | "
              f"Mass: {mass_acc/num_batches:.5f} | "
              f"Geo: {geo_acc/num_batches:.5f}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'edge_index': edge_index.cpu(),
        'stats': dataset.stats,
        'var_names': dataset.var_names
    }, checkpoint_path)

    print(f"\n[AIDA TRAINING] Checkpoint written to: '{checkpoint_path}'")


def main():
    parser = argparse.ArgumentParser(description="Train AIDA GNN surrogate model on log-state Zarr store.")
    parser.add_argument("-z", "--zarr", default="../data/icosahedral_2023_logstate.zarr", help="Path to input log-state .zarr store")
    parser.add_argument("-g", "--edges", default="../data/graph/icosahedral_edge_index_m4.pt", help="Path to PyTorch edge_index tensor")
    parser.add_argument("-c", "--checkpoint", default="checkpoints/aida_gnn_surrogate_logstate.pt", help="Output model path")
    parser.add_argument("-e", "--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--lr", type=float, default=0.0003, help="Learning rate")

    args = parser.parse_args()
    run_training(args.zarr, args.edges, args.checkpoint, args.epochs, args.batch_size, args.lr)

if __name__ == "__main__":
    main()
