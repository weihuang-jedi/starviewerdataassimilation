#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
Trains the AIDA GNN surrogate model on scale-invariant log-state Zarr stores
[ln_T, u, v, w, q, ln_rho, ln_p] using Non-Hydrostatic Icosahedral Loss 
and Pressure Regularization (Laplacian Smoothness + Asymmetric Low Penalty).
"""

import os
import argparse
import numpy as np
import zarr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing


LOG_STATE_VARS = [
    'ln_t_icosahedral',    # Index 0
    'u_icosahedral',       # Index 1
    'v_icosahedral',       # Index 2
    'w_icosahedral',       # Index 3
    'q_icosahedral',       # Index 4
    'ln_rho_icosahedral',  # Index 5
    'ln_p_icosahedral'     # Index 6
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

        first_var = self.root[self.var_names[0]]
        self.total_timesteps, self.num_levels, self.num_nodes = first_var.shape

        print("========================================================")
        print("[AIDA DATASET] Initialized Log-State Zarr Loader")
        print(f"Store Path   : {zarr_path}")
        print(f"Variables   : {self.var_names}")
        print(f"Timesteps   : {self.total_timesteps}")
        print(f"Mesh Layout : {self.num_levels} height levels x {self.num_nodes} nodes")
        print("========================================================\n")

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
# 3. COMBINED NON-HYDROSTATIC & REGULARIZED LOSS
# ==========================================
class AIDAPressureRegularizedLoss(nn.Module):
    def __init__(
        self,
        means: dict,
        stds: dict,
        p_idx: int = 6,                  # Index 6 = ln_p_icosahedral
        tau_min_p: float = -3.5,         # Standardized threshold for extreme low pressure
        lambda_smooth_p: float = 0.08,   # Eliminates spatial checkerboard noise
        lambda_asym_p: float = 0.5,      # Suppresses runaway low pressure
        weight_mse: float = 1.0,
        weight_state_eq: float = 0.5,
        weight_mass: float = 0.0,
        weight_geo: float = 0.0,
        R_d: float = 287.058
    ):
        super().__init__()
        self.p_idx = p_idx
        self.tau_min_p = tau_min_p
        self.lambda_smooth_p = lambda_smooth_p
        self.lambda_asym_p = lambda_asym_p
        self.weight_mse = weight_mse
        self.weight_state_eq = weight_state_eq
        self.weight_mass = weight_mass
        self.weight_geo = weight_geo
        self.R_d = R_d
        self.ln_R_d = np.log(R_d)

        self.register_buffer("mu_ln_t", torch.tensor(means['ln_t_icosahedral']['mean'], dtype=torch.float32))
        self.register_buffer("std_ln_t", torch.tensor(means['ln_t_icosahedral']['std'], dtype=torch.float32))

        self.register_buffer("mu_ln_rho", torch.tensor(means['ln_rho_icosahedral']['mean'], dtype=torch.float32))
        self.register_buffer("std_ln_rho", torch.tensor(means['ln_rho_icosahedral']['std'], dtype=torch.float32))

        self.register_buffer("mu_ln_p", torch.tensor(means['ln_p_icosahedral']['mean'], dtype=torch.float32))
        self.register_buffer("std_ln_p", torch.tensor(means['ln_p_icosahedral']['std'], dtype=torch.float32))

        self.mse_fn = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, edge_index: torch.Tensor) -> dict:
        # 1. Base MSE Loss
        loss_mse = self.mse_fn(pred, target)

        # 2. Graph Laplacian Smoothness Loss on ln_p (Prevents Checkerboard Noise)
        # Shape: [Batch, Vars, Levels, Nodes] -> pred[:, 6, :, :]
        p_pred = pred[:, self.p_idx, :, :]
        src, dst = edge_index[0], edge_index[1]
        diff_p = p_pred[:, :, src] - p_pred[:, :, dst]
        loss_smooth_p = torch.mean(torch.square(diff_p))

        # 3. Asymmetric Low-Pressure Penalty (Prevents Extreme Low Pressure)
        # Activates when standardized ln_p prediction drops below tau_min_p
        violation = F.relu(self.tau_min_p - p_pred)
        loss_asym_p = torch.mean(torch.square(violation))

        # 4. Ideal Gas State Residual
        ln_T_phys   = torch.clamp(pred[:, 0, :, :] * self.std_ln_t + self.mu_ln_t, min=4.95, max=6.0)
        ln_rho_phys = torch.clamp(pred[:, 5, :, :] * self.std_ln_rho + self.mu_ln_rho, min=-15.0, max=2.0)
        ln_p_phys   = torch.clamp(pred[:, 6, :, :] * self.std_ln_p + self.mu_ln_p, min=-5.0, max=13.0)

        state_eq_residual = ln_p_phys - (ln_rho_phys + self.ln_R_d + ln_T_phys)
        loss_state_eq = torch.mean(torch.abs(state_eq_residual))

        # 5. Global Mass Conservation Penalty (Density Integral Stability)
        loss_mass = torch.mean(torch.abs(torch.mean(pred[:, 5, :, :], dim=-1)))

        # 6. Geostrophic Balance Loss (Spatial Gradients over Graph Edges)
        u_pred = pred[:, 1, :, :]
        v_pred = pred[:, 2, :, :]
        src, dst = edge_index[0], edge_index[1]

        # Spatial gradient proxy on graph topology
        dp_edge = torch.abs(ln_p_phys[:, :, src] - ln_p_phys[:, :, dst])
        wind_mag = torch.sqrt(u_pred[:, :, src]**2 + v_pred[:, :, src]**2 + 1e-6)
        loss_geo = torch.mean(torch.abs(dp_edge - wind_mag * 0.1))

        total_loss = (
            self.weight_mse * loss_mse +
            self.weight_state_eq * loss_state_eq +
            self.weight_mass * loss_mass +
            self.weight_geo * loss_geo +
            self.lambda_smooth_p * loss_smooth_p +
            self.lambda_asym_p * loss_asym_p
        )

        return {
            "loss": total_loss,
            "mse": loss_mse,
            "state_eq": loss_state_eq,
            "smooth_p": loss_smooth_p,
            "asym_p": loss_asym_p
        }


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
def run_training(zarr_path, edge_file, checkpoint_path, epochs, batch_size, lr, tau_min_p, lambda_smooth_p, lambda_asym_p):
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

    # Initialize loss with calibrated pressure parameters
    criterion = AIDAPressureRegularizedLoss(
        means=dataset.stats,
        stds=dataset.stats,
        p_idx=6,                       # Index 6 corresponds to ln_p
        tau_min_p=tau_min_p,
        lambda_smooth_p=lambda_smooth_p,
        lambda_asym_p=lambda_asym_p
    ).to(device)

    print(f"[AIDA TRAINING] Starting execution for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, mse_acc, state_acc, smooth_acc, asym_acc = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)

            loss_dict = criterion(y_pred, y_batch, edge_index)
            total_loss = loss_dict["loss"]

            if torch.isnan(total_loss):
                print(f"[FATAL] Loss became NaN at batch {batch_idx}. Skipping step.")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tot_loss += total_loss.item()
            mse_acc += loss_dict["mse"].item()
            state_acc += loss_dict["state_eq"].item()
            smooth_acc += loss_dict["smooth_p"].item()
            asym_acc += loss_dict["asym_p"].item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] Batch {batch_idx:04d} | "
                    f"Total: {total_loss.item():.4f} | "
                    f"MSE: {loss_dict['mse'].item():.4f} | "
                    f"SmoothP: {loss_dict['smooth_p'].item():.4f} | "
                    f"AsymP: {loss_dict['asym_p'].item():.4f}"
                )

        num_batches = max(1, len(dataloader))
        print(f"\n---> Epoch [{epoch:02d}/{epochs:02d}] Summary | "
              f"Total Loss: {tot_loss/num_batches:.5f} | "
              f"MSE: {mse_acc/num_batches:.5f} | "
              f"StateEq: {state_acc/num_batches:.5f} | "
              f"SmoothP: {smooth_acc/num_batches:.5f} | "
              f"AsymP: {asym_acc/num_batches:.5f}\n")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'edge_index': edge_index.cpu(),
        'stats': dataset.stats,
        'var_names': dataset.var_names
    }, checkpoint_path)

    print(f"[AIDA TRAINING] Checkpoint written to: '{checkpoint_path}'")


def main():
    parser = argparse.ArgumentParser(description="Train AIDA GNN surrogate model on log-state Zarr store.")
    parser.add_argument("-z", "--zarr", default="../data/icosahedral_2023_logstate.zarr", help="Path to input log-state .zarr store")
    parser.add_argument("-g", "--edges", default="../data/graph/icosahedral_edge_index_m4.pt", help="Path to PyTorch edge_index tensor")
    parser.add_argument("-c", "--checkpoint", default="checkpoints/aida_gnn_surrogate_logstate.pt", help="Output model path")
    parser.add_argument("-e", "--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--tau_min_p", type=float, default=-3.5, help="Threshold below which asymmetric penalty activates")
    parser.add_argument("--lambda_smooth_p", type=float, default=0.08, help="Pressure smoothness loss weight")
    parser.add_argument("--lambda_asym_p", type=float, default=0.5, help="Asymmetric pressure penalty weight")

    args = parser.parse_args()
    run_training(
        args.zarr, args.edges, args.checkpoint, 
        args.epochs, args.batch_size, args.lr,
        args.tau_min_p, args.lambda_smooth_p, args.lambda_asym_p
    )


if __name__ == "__main__":
    main()
