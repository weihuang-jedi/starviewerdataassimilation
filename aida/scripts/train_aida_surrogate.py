#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
Trains the Icosahedral Graph Neural Network (GNN) surrogate model.
Applies Z-score normalization and DA-aware loss weighting.
"""

import os
import glob
import time
import argparse
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
STATE_VARS = ["p", "t", "u", "v", "q"]

# Physical Mean & Standard Deviation for Z-Score Normalization
# Shapes aligned to (1, 5, 1, 1) for broadcasting over (Batch, Vars, Levels, Nodes)
VAR_MEAN_VALS = [101325.0, 250.0, 0.0, 0.0, 0.003]
VAR_STD_VALS  = [15000.0,  35.0, 15.0, 15.0, 0.004]

# ==========================================
# GNN MODEL DEFINITION
# ==========================================
class IcosahedralGraphGNN(nn.Module):
    """
    Message Passing GNN for icosahedral grid forecasts.
    Expects state vector of shape: (B, 5, 32, 2562)
    Flattened channel dimension: 5 vars * 32 levels = 160
    """
    def __init__(self, edge_index, in_channels=160, hidden_dim=128, out_channels=160):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        
        # Node Feature Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message Passing Layer 1
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message Passing Layer 2
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node Output Decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x):
        # x shape: (B, C=5, L=32, N=2562)
        b, c, l, n = x.shape
        x_flat = x.view(b, c * l, n).transpose(1, 2)  # (B, N, C*L=160)
        
        h = self.node_encoder(x_flat)
        src, dst = self.edge_index[0], self.edge_index[1]
        
        # Message passing round 1
        msg1 = torch.cat([h[:, src, :], h[:, dst, :]], dim=-1)
        out1 = torch.zeros_like(h)
        out1.index_add_(1, dst, self.conv1(msg1))
        h = h + out1
        
        # Message passing round 2
        msg2 = torch.cat([h[:, src, :], h[:, dst, :]], dim=-1)
        out2 = torch.zeros_like(h)
        out2.index_add_(1, dst, self.conv2(msg2))
        h = h + out2
        
        # Decode residual tendency
        out = self.node_decoder(h)
        out = out.transpose(1, 2).view(b, c, l, n)
        
        # Residual step prediction
        return x + out

# ==========================================
# DA-AWARE LOSS MODULE
# ==========================================
class DAAtmosphericLoss(nn.Module):
    def __init__(self, bmatrix_path=None, device="cpu"):
        super().__init__()
        self.device = device
        
        # Priority multipliers in normalized space
        self.multipliers = {
            "p": 1.0,
            "t": 2.0,
            "u": 5.0,
            "v": 5.0,
            "q": 8.0
        }
        
        # Default level weights (1, 32, 1)
        self.level_weights = {
            var: torch.ones((1, 32, 1), device=device) for var in STATE_VARS
        }
        
        if bmatrix_path and os.path.exists(bmatrix_path):
            try:
                ds_b = xr.open_dataset(bmatrix_path)
                for var in STATE_VARS:
                    b_key = next((k for k in [f"{var}_var", var, f"{var}_err"] if k in ds_b), None)
                    if b_key:
                        var_data = ds_b[b_key].values
                        while var_data.ndim > 1:
                            var_data = np.mean(var_data, axis=-1)
                        if len(var_data) == 32:
                            inv_var = 1.0 / np.maximum(var_data, 1e-6)
                            inv_var = inv_var / np.mean(inv_var)  # Scale normalized
                            self.level_weights[var] = torch.tensor(
                                inv_var, dtype=torch.float32, device=device
                            ).view(1, 32, 1)
            except Exception as e:
                print(f"[WARNING] Could not parse B-matrix level variance: {e}")

    def forward(self, pred_norm, target_norm):
        total_loss = 0.0
        loss_dict = {}
        for idx, var in enumerate(STATE_VARS):
            p_var = pred_norm[:, idx, :, :]
            t_var = target_norm[:, idx, :, :]
            diff_sq = (p_var - t_var) ** 2
            weighted_mse = torch.mean(diff_sq * self.level_weights[var])
            var_loss = self.multipliers[var] * weighted_mse
            loss_dict[var] = var_loss.item()
            total_loss += var_loss
        return total_loss, loss_dict

# ==========================================
# DATASET
# ==========================================
class IcosahedralPairedDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "global_icosahedral_m4.*.nc")))
        if len(self.files) < 2:
            raise ValueError(f"Need at least 2 NetCDF files in {data_dir} for step pairing.")

    def __len__(self):
        return len(self.files) - 1

    def __getitem__(self, idx):
        ds_t0 = xr.open_dataset(self.files[idx])
        ds_t1 = xr.open_dataset(self.files[idx + 1])
        
        x0 = np.stack([ds_t0[v].values for v in STATE_VARS], axis=0) # (5, 32, 2562)
        x1 = np.stack([ds_t1[v].values for v in STATE_VARS], axis=0) # (5, 32, 2562)
        
        ds_t0.close()
        ds_t1.close()
        return torch.tensor(x0, dtype=torch.float32), torch.tensor(x1, dtype=torch.float32)

# ==========================================
# MAIN TRAINING ROUTINE
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/nc")
    parser.add_argument("--graph_path", type=str, default="../data/graph/edge_index_m4.pt")
    parser.add_argument("--bmatrix_path", type=str, default="../bmatrix/bmatrix_from_gfs_analysis.nc")
    parser.add_argument("--ckpt_out", type=str, default="checkpoints/aida_gnn_v1.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Graph Topology
    edge_index = torch.load(args.graph_path, map_location=device)
    model = IcosahedralGraphGNN(edge_index=edge_index).to(device)
    
    # Dataset & Loader
    dataset = IcosahedralPairedDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Loss & Optimizer
    criterion = DAAtmosphericLoss(args.bmatrix_path, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Normalization Tensors
    var_mean = torch.tensor(VAR_MEAN_VALS, device=device).view(1, 5, 1, 1)
    var_std  = torch.tensor(VAR_STD_VALS, device=device).view(1, 5, 1, 1)

    print(f"[TRAIN] Starting training on {len(dataset)} pairs across {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for x0, x1 in dataloader:
            x0, x1 = x0.to(device), x1.to(device)
            
            # Z-Score Normalization
            x0_norm = (x0 - var_mean) / var_std
            x1_norm = (x1 - var_mean) / var_std
            
            optimizer.zero_grad()
            pred_norm = model(x0_norm)
            
            loss, _ = criterion(pred_norm, x1_norm)
            loss.backward()
            
            # Gradient clipping to ensure stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.6f}")

    # Save State Dict
    torch.save(model.state_dict(), args.ckpt_out)
    print(f"[SAVE] Saved surrogate weights to '{args.ckpt_out}' successfully.")

if __name__ == "__main__":
    main()
