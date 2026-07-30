#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
Trains the AIDA GNN atmospheric surrogate model directly on native .zarr stores.
Includes automated NaN sanitization, robust Z-score normalization, and stable PINN losses.
"""

import os
import argparse
import numpy as np
import zarr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing

# Primary variable mapping tuned to icosahedral_2023.zarr keys
TARGET_STATE_VARS = [
    't_icosahedral',
    'u_icosahedral',
    'v_icosahedral',
    'w_icosahedral',
    'q_icosahedral',
    'p_icosahedral'
]

DEFAULT_VAR_ALIASES = {
    't': ['t_icosahedral', 't', 'ta', 'temperature', 'TMP'],
    'u': ['u_icosahedral', 'u', 'ua', 'u_wind', 'UGRD'],
    'v': ['v_icosahedral', 'v', 'va', 'v_wind', 'VGRD'],
    'w': ['w_icosahedral', 'w', 'wa', 'dzdt', 'VVEL'],
    'q': ['q_icosahedral', 'q', 'hus', 'specific_humidity', 'SPFH'],
    'p': ['p_icosahedral', 'p', 'pres', 'pressure', 'PRES']
}

# ==========================================
# 1. GRAPH MESSAGE PASSING LAYER
# ==========================================
class IcosahedralGraphConv(MessagePassing):
    """
    Graph Convolutional Layer passing messages along icosahedral mesh edges.
    """
    def __init__(self, in_channels, out_channels):
        super(IcosahedralGraphConv, self).__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=self.lin(x))

    def message(self, x_j):
        return x_j


# ==========================================
# 2. NATIVE ZARR DATASET LOADER
# ==========================================
class ZarrIcosahedralDataset(Dataset):
    """
    Direct Zarr dataset loader for icosahedral grid stores.
    Reads native shape: (Time, Height, Node) per variable.
    """
    def __init__(self, zarr_path, var_names=None, sequence_len=1):
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr store not found at: {zarr_path}")

        self.root = zarr.open(zarr_path, mode='r')
        available_keys = list(self.root.keys())

        # Resolve requested or aliased variable names
        self.var_names = self._resolve_variable_keys(var_names, available_keys)
        self.sequence_len = sequence_len

        # Probe dimensions using first resolved variable
        first_var = self.root[self.var_names[0]]

        if first_var.ndim == 3:
            self.total_timesteps, self.num_levels, self.num_nodes = first_var.shape
        elif first_var.ndim == 2:
            self.total_timesteps, self.num_nodes = first_var.shape
            self.num_levels = 1
        else:
            raise ValueError(f"Unexpected shape for variable '{self.var_names[0]}': {first_var.shape}")

        print("========================================================")
        print("[AIDA DATASET] Initialized Native Zarr Graph Loader")
        print(f"Store Path  : {zarr_path}")
        print(f"Variables   : {self.var_names}")
        print(f"Timesteps   : {self.total_timesteps}")
        print(f"Mesh Layout : {self.num_levels} height levels x {self.num_nodes} nodes")
        print("========================================================\n")

        # Compute robust Z-score normalization statistics
        self.stats = {}
        for var in self.var_names:
            arr = self.root[var]
            
            # Read first 100 timesteps to get valid baseline
            sample_data = np.asarray(arr[:min(100, self.total_timesteps)])
            sample_data = sample_data[np.isfinite(sample_data)]
            
            if sample_data.size == 0:
                mean, std = 0.0, 1.0
            else:
                mean = float(np.mean(sample_data))
                std = float(np.std(sample_data))
            
            if std < 1e-6 or np.isnan(std):
                std = 1.0
            if np.isnan(mean):
                mean = 0.0

            self.stats[var] = {'mean': mean, 'std': std}
            print(f"  -> [{var}] Mean: {mean:.4f} | Std: {std:.4f}")

    def _resolve_variable_keys(self, user_vars, available_keys):
        target_keys = user_vars if user_vars else TARGET_STATE_VARS
        resolved, missing = [], []

        for v in target_keys:
            if v in available_keys:
                resolved.append(v)
            else:
                found_alias = None
                if v in DEFAULT_VAR_ALIASES:
                    for alias in DEFAULT_VAR_ALIASES[v]:
                        if alias in available_keys:
                            found_alias = alias
                            break
                if found_alias:
                    resolved.append(found_alias)
                else:
                    missing.append(v)

        if missing:
            raise KeyError(f"Missing keys {missing} in Zarr store '{zarr_path}'")

        return resolved

    def __len__(self):
        return self.total_timesteps - self.sequence_len

    def __getitem__(self, idx):
        x_vars, y_vars = [], []

        for var in self.var_names:
            arr = self.root[var]
            mean = self.stats[var]['mean']
            std  = self.stats[var]['std']

            # Extract raw arrays
            raw_x = np.asarray(arr[idx], dtype=np.float32)
            raw_y = np.asarray(arr[idx + self.sequence_len], dtype=np.float32)

            # 1. Clean missing/fill values (replace NaNs with mean value prior to norm)
            raw_x = np.nan_to_num(raw_x, nan=mean, posinf=mean, neginf=mean)
            raw_y = np.nan_to_num(raw_y, nan=mean, posinf=mean, neginf=mean)

            # 2. Perform Z-score normalization
            data_x = (raw_x - mean) / std
            data_y = (raw_y - mean) / std

            if data_x.ndim == 1:
                data_x = data_x[np.newaxis, :]
                data_y = data_y[np.newaxis, :]

            x_vars.append(data_x)
            y_vars.append(data_y)

        # Shape: (Vars, Levels, Nodes)
        x_tensor = torch.tensor(np.stack(x_vars, axis=0), dtype=torch.float32)
        y_tensor = torch.tensor(np.stack(y_vars, axis=0), dtype=torch.float32)

        # Safeguard float precision outputs
        x_tensor = torch.nan_to_num(x_tensor, nan=0.0)
        y_tensor = torch.nan_to_num(y_tensor, nan=0.0)

        return x_tensor, y_tensor


# ==========================================
# 3. ATMOSPHERIC PINN LOSS FUNCTION
# ==========================================
class AtmosphericPINNLoss(nn.Module):
    """
    PINN Atmospheric Loss with clamped stability thresholds.
    """
    def __init__(self, lambda_mse=1.0, lambda_hydro=0.05, lambda_kinetic=0.1, eps=1e-2):
        super(AtmosphericPINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_mse = lambda_mse
        self.lambda_hydro = lambda_hydro
        self.lambda_kinetic = lambda_kinetic
        self.eps = eps

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)

        num_vars = pred.shape[1]
        num_levels = pred.shape[2]

        # Hydrostatic Balance (t_icosahedral index 0, p_icosahedral index 5)
        if num_vars >= 6 and num_levels > 1:
            p_pred = pred[:, 5, :, :]
            t_pred = pred[:, 0, :, :]
            dp = p_pred[:, 1:, :] - p_pred[:, :-1, :]
            denom = torch.clamp(torch.abs(t_pred[:, 1:, :]), min=self.eps)
            loss_hydro = torch.mean(torch.abs(dp / denom))
        else:
            loss_hydro = torch.tensor(0.0, device=pred.device)

        # Kinetic smoothness penalty (u index 1, v index 2)
        if num_vars >= 3:
            u_pred = pred[:, 1, :, :]
            v_pred = pred[:, 2, :, :]
            loss_kinetic = torch.mean(torch.abs(u_pred[:, :, 1:] - u_pred[:, :, :-1])) + \
                           torch.mean(torch.abs(v_pred[:, :, 1:] - v_pred[:, :, :-1]))
        else:
            loss_kinetic = torch.tensor(0.0, device=pred.device)

        total_loss = (self.lambda_mse * loss_mse) + \
                     (self.lambda_hydro * loss_hydro) + \
                     (self.lambda_kinetic * loss_kinetic)

        return total_loss, loss_mse, loss_hydro, loss_kinetic


# ==========================================
# 4. ICOSAHEDRAL GNN SURROGATE MODEL
# ==========================================
class IcosahedralGNNSurrogate(nn.Module):
    def __init__(self, edge_index, in_vars=6, levels=32, hidden_dim=128):
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


# ==========================================
# 5. MESH EDGE HELPER
# ==========================================
def generate_or_load_edge_index(num_nodes=2562, edge_file=None):
    if edge_file and os.path.exists(edge_file):
        print(f"[AIDA GRAPH] Loading edge topology from: {edge_file}")
        edge_index = torch.load(edge_file, weights_only=False)
    else:
        print(f"[AIDA GRAPH] Generating fallback graph topology for {num_nodes} nodes...")
        src = torch.arange(num_nodes)
        dst = (src + 1) % num_nodes
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    return edge_index


# ==========================================
# 6. TRAINING EXECUTOR
# ==========================================
def run_training(zarr_path, edge_file, checkpoint_path, epochs, batch_size, lr, var_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[AIDA TRAINING] Execution Device: {device}")

    dataset = ZarrIcosahedralDataset(zarr_path=zarr_path, var_names=var_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    edge_index = generate_or_load_edge_index(num_nodes=dataset.num_nodes, edge_file=edge_file).to(device)

    model = IcosahedralGNNSurrogate(edge_index=edge_index, in_vars=len(dataset.var_names), levels=dataset.num_levels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = AtmosphericPINNLoss()

    print(f"[AIDA TRAINING] Starting execution for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_accum, mse_accum, hydro_accum, kinetic_accum = 0.0, 0.0, 0.0, 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                raise ValueError(f"NaN detected in raw input tensor batch {batch_idx}! Check Zarr store normalization.")

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)

            total_loss, loss_mse, loss_hydro, loss_kinetic = criterion(y_pred, y_batch)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_accum += total_loss.item()
            mse_accum += loss_mse.item()
            hydro_accum += loss_hydro.item()
            kinetic_accum += loss_kinetic.item()

        num_batches = len(dataloader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Total: {total_loss_accum/num_batches:.6f} | "
              f"MSE: {mse_accum/num_batches:.6f} | "
              f"Hydro: {hydro_accum/num_batches:.6f} | "
              f"Kinetic: {kinetic_accum/num_batches:.6f}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'edge_index': edge_index.cpu(),
        'stats': dataset.stats,
        'var_names': dataset.var_names
    }, checkpoint_path)

    print(f"\n[AIDA TRAINING] Checkpoint successfully written to: '{checkpoint_path}'")


def main():
    parser = argparse.ArgumentParser(description="Train AIDA GNN surrogate model on native Zarr stores.")
    parser.add_argument("-z", "--zarr", required=True, help="Path to input .zarr store")
    parser.add_argument("-g", "--edges", default=None, help="Path to PyTorch edge_index tensor (.pt file)")
    parser.add_argument("-c", "--checkpoint", default="checkpoints/aida_gnn_surrogate.pt", help="Output model path")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-v", "--var_names", nargs="+", default=None, help="Explicit list of variable names in Zarr store")

    args = parser.parse_args()
    run_training(args.zarr, args.edges, args.checkpoint, args.epochs, args.batch_size, args.lr, args.var_names)

if __name__ == "__main__":
    main()
