#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
AIDA GNN Surrogate Model Training Script for Icosahedral Atmospheric Grids.

Fixes & Enhancements:
  - Saved checkpoint dictionary explicitly includes 'stats' for cycling inference.
  - LayerNorm in message passing blocks to prevent gradient explosion.
  - Safe node degree normalization in Laplacian penalty to eliminate NaN sources.
  - Tuned default pressure Laplacian weight (lambda_laplacian_p) to suppress checkerboards.
  - Protected thermodynamic coupling and gradient matching loss terms.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


# =============================================================================
# GLOBAL STATE VARIABLE DEFINITIONS
# =============================================================================
LOG_STATE_VARS = [
    'ln_t_icosahedral',
    'u_icosahedral',
    'v_icosahedral',
    'w_icosahedral',
    'q_icosahedral',
    'ln_rho_icosahedral',
    'ln_p_icosahedral'
]


# =============================================================================
# 1. DATASET MODULES (ZARR & SYNTHETIC FALLBACK)
# =============================================================================
class LogStateZarrDataset(Dataset):
    """
    Dataset wrapper for multi-variable Zarr stores with separate 3D arrays:
    Keys: ['ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
           'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral']
    Array shape per variable: [Time=1460, Levels=32, Nodes=2562]
    Output shape per sample:   [Vars=7, Levels=32, Nodes=2562]
    """
    def __init__(self, zarr_path: str):
        super().__init__()
        self.zarr_path = zarr_path

        # Use module-level variable definition
        self.var_keys = LOG_STATE_VARS

        try:
            import zarr
        except ImportError:
            raise ImportError("zarr library is required. Run 'pip install zarr'.")

        self.root = zarr.open(zarr_path, mode='r')

        available_keys = list(self.root.array_keys())
        for k in self.var_keys:
            if k not in available_keys:
                raise KeyError(f"Expected key '{k}' not found in Zarr store at '{zarr_path}'. Found: {available_keys}")

        first_arr = self.root[self.var_keys[0]]
        self.num_time_steps = first_arr.shape[0] - 1  # t -> t+1 pairs
        self.num_vars = len(self.var_keys)
        self.num_levels = first_arr.shape[1]  # 32
        self.num_nodes = first_arr.shape[2]   # 2562

        print(f"[DATASET] Loaded Multi-Array Zarr dataset from '{zarr_path}'")
        print(f"          Variables ({self.num_vars}): {self.var_keys}")
        print(f"          Dimensions: Time={self.num_time_steps + 1}, Levels={self.num_levels}, Nodes={self.num_nodes}")

    def __len__(self):
        return self.num_time_steps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_list = [np.array(self.root[key][idx], dtype=np.float32) for key in self.var_keys]
        y_list = [np.array(self.root[key][idx + 1], dtype=np.float32) for key in self.var_keys]

        x = np.stack(x_list, axis=0)
        y = np.stack(y_list, axis=0)

        # Basic NaN safeguard on data read
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        return torch.from_numpy(x), torch.from_numpy(y)


class SyntheticAIDAStateDataset(Dataset):
    """Fallback dataset simulating log-state atmospheric variables on mesh."""
    def __init__(self, num_samples: int = 80, num_nodes: int = 2562, num_levels: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_levels = num_levels
        self.num_vars = len(LOG_STATE_VARS)

        np.random.seed(42)
        self.data_x = np.random.randn(num_samples, 7, num_levels, num_nodes).astype(np.float32)
        self.data_y = self.data_x * 0.98 + 0.02 * np.random.randn(num_samples, 7, num_levels, num_nodes).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.data_x[idx]), torch.from_numpy(self.data_y[idx])


# =============================================================================
# 2. GRAPH TOPOLOGY GENERATION
# =============================================================================
def generate_or_load_edge_index(num_nodes: int, edge_file: str = "") -> torch.Tensor:
    """Loads existing edge connectivity or constructs a synthetic k-NN edge graph."""
    if edge_file and os.path.exists(edge_file):
        print(f"[GRAPH] Loading precomputed edge topology from '{edge_file}'...")
        edge_index = torch.load(edge_file)
        if isinstance(edge_index, dict) and "edge_index" in edge_index:
            edge_index = edge_index["edge_index"]
        return edge_index.to(torch.long)

    print(f"[GRAPH] Generating synthetic icosahedral mesh graph for {num_nodes} nodes...")
    phi = np.linspace(0, np.pi, int(np.sqrt(num_nodes)))
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_nodes)))
    phi_m, theta_m = np.meshgrid(phi, theta)

    x = np.sin(phi_m) * np.cos(theta_m)
    y = np.sin(phi_m) * np.sin(theta_m)
    z = np.cos(phi_m)
    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T[:num_nodes]

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=7)

    src_list, dst_list = [], []
    for i, neighbors in enumerate(indices):
        for n in neighbors[1:]:
            src_list.append(i)
            dst_list.append(n)

    return torch.tensor([src_list, dst_list], dtype=torch.long)


# =============================================================================
# 3. GNN SURROGATE ARCHITECTURE (WITH LAYER NORM STABILITY)
# =============================================================================
class GraphConvBlock(nn.Module):
    """Graph Convolution Message Passing Block with LayerNorm for numerical stability."""
    def __init__(self, channels: int):
        super().__init__()
        self.fc_msg = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        self.fc_update = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, C, L, N = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(B * L * N, C)

        src, dst = edge_index[0], edge_index[1]

        shift = torch.arange(B * L, device=x.device).unsqueeze(1) * N
        src_expanded = (src.unsqueeze(0) + shift).reshape(-1)
        dst_expanded = (dst.unsqueeze(0) + shift).reshape(-1)

        msg_in = torch.cat([x_perm[src_expanded], x_perm[dst_expanded]], dim=-1)
        messages = self.fc_msg(msg_in)

        aggr_msg = torch.zeros_like(x_perm)
        aggr_msg.index_add_(0, dst_expanded, messages)

        # Compute degree average for aggregation
        deg = torch.zeros(B * L * N, 1, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst_expanded, torch.ones((dst_expanded.shape[0], 1), device=x.device, dtype=x.dtype))
        aggr_msg = aggr_msg / torch.clamp(deg, min=1.0)

        updated = self.fc_update(torch.cat([x_perm, aggr_msg], dim=-1))
        updated = self.norm(updated)

        out = updated.reshape(B, L, N, C).permute(0, 3, 1, 2)
        return x + out


class IcosahedralGNNSurrogate(nn.Module):
    """GNN Atmospheric Surrogate Model for icosahedral mesh fields."""
    def __init__(self, in_vars: int = 7, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.encoder = nn.Conv2d(in_vars, hidden_dim, kernel_size=1)
        self.gnn_layers = nn.ModuleList([GraphConvBlock(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Conv2d(hidden_dim, in_vars, kernel_size=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for layer in self.gnn_layers:
            h = layer(h, edge_index)
        out = self.decoder(h)
        return out


# =============================================================================
# 4. REGULARIZED SURROGATE LOSS MODULE (STABILIZED)
# =============================================================================
# =============================================================================
# 4. REGULARIZED SURROGATE LOSS MODULE (STABILIZED + Q-SCALE FIXED)
# =============================================================================
class AIDASurrogateLoss(nn.Module):
    """
    Physically-constrained regularized loss module for AIDA GNN Surrogate Model.
    Includes log-scale loss for specific humidity (q) to fix relative error explosion.
    """
    def __init__(
        self,
        p_idx: int = 6,
        q_idx: int = 4,
        sharp_var_indices: list[int] = [0, 1, 2, 4, 6],  # T, u, v, q, p
        weight_mse: float = 1.0,
        lambda_laplacian_p: float = 0.18,
        weight_grad_state: float = 0.20,
        lambda_asym_p: float = 0.25,
        weight_state_eq: float = 0.10,
        weight_q_log: float = 0.15,                      # Weight for q log-scale matching
        weight_joint_bias: float = 0.05,                  # Mild joint bias stabilizer
        tau_min_p: float = -0.2894,
        mu_ln_t: float = 5.5, std_ln_t: float = 0.2,
        mu_ln_rho: float = -0.5, std_ln_rho: float = 0.5,
        mu_ln_p: float = 11.5, std_ln_p: float = 0.3,
        R_d: float = 287.058
    ):
        super().__init__()
        self.p_idx = p_idx
        self.q_idx = q_idx
        self.sharp_var_indices = sharp_var_indices

        self.weight_mse = weight_mse
        self.lambda_laplacian_p = lambda_laplacian_p
        self.weight_grad_state = weight_grad_state
        self.lambda_asym_p = lambda_asym_p
        self.weight_state_eq = weight_state_eq
        self.weight_q_log = weight_q_log
        self.weight_joint_bias = weight_joint_bias
        self.tau_min_p = tau_min_p

        self.mu_ln_t, self.std_ln_t = mu_ln_t, std_ln_t
        self.mu_ln_rho, self.std_ln_rho = mu_ln_rho, std_ln_rho
        self.mu_ln_p, self.std_ln_p = mu_ln_p, std_ln_p
        self.register_buffer("ln_R_d", torch.log(torch.tensor(R_d)))

        self.mse_fn = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:

        B, V, L, N = pred.shape
        src, dst = edge_index[0], edge_index[1]

        # 1. Base Data Fidelity Loss (MSE)
        loss_mse = self.mse_fn(pred, target)

        # 2. 2nd-Order Graph Laplacian Pressure Penalty (Degree-Normalized)
        p_pred = pred[:, self.p_idx, :, :]  # [B, L, N]

        p_neighbor_sum = torch.zeros_like(p_pred)
        dst_expanded = dst.view(1, 1, -1).expand(B, L, -1)
        p_neighbor_sum.scatter_add_(2, dst_expanded, p_pred.index_select(2, src))

        # Safe degree calculation to avoid 0 division
        deg = torch.zeros(N, device=p_pred.device, dtype=p_pred.dtype)
        deg.index_add_(0, dst, torch.ones_like(src, dtype=p_pred.dtype))
        deg = deg.view(1, 1, N)
        deg_clamped = torch.clamp(deg, min=1.0)

        p_neighbor_avg = p_neighbor_sum / deg_clamped
        loss_laplacian_p = torch.mean(torch.square(p_pred - p_neighbor_avg))

        # 3. State Gradient Matching Loss (Preserves Fronts in T, u, v, q, p)
        sharp_pred = pred[:, self.sharp_var_indices, :, :]
        sharp_target = target[:, self.sharp_var_indices, :, :]

        diff_pred = sharp_pred.index_select(3, src) - sharp_pred.index_select(3, dst)
        diff_target = sharp_target.index_select(3, src) - sharp_target.index_select(3, dst)
        loss_grad_state = torch.mean(torch.abs(diff_pred - diff_target))

        # 4. Asymmetric Barrier Penalty for Low Pressure Spikes
        violation = F.relu(self.tau_min_p - p_pred)
        loss_asym_p = torch.mean(torch.square(violation))

        # 5. Ideal Gas Thermodynamic Coupling Residual
        ln_T_phys = torch.clamp(pred[:, 0, :, :] * self.std_ln_t + self.mu_ln_t, min=4.95, max=6.0)
        ln_rho_phys = torch.clamp(pred[:, 5, :, :] * self.std_ln_rho + self.mu_ln_rho, min=-15.0, max=2.0)
        ln_p_phys = torch.clamp(pred[:, 6, :, :] * self.std_ln_p + self.mu_ln_p, min=-5.0, max=13.0)

        state_eq_residual = ln_p_phys - (ln_rho_phys + self.ln_R_d + ln_T_phys)
        loss_state_eq = torch.mean(torch.abs(state_eq_residual))

        # 6. Specific Humidity (q) Log-Space Scale Penalty (Fixes Stratospheric Relative Error)
        q_pred = pred[:, self.q_idx, :, :]
        q_target = target[:, self.q_idx, :, :]
        # Offset epsilon ensures safe log transform across all 32 pressure levels
        eps = 1e-6
        loss_q_log = torch.mean(torch.abs(torch.log(F.relu(q_pred) + eps) - torch.log(F.relu(q_target) + eps)))

        # 7. Joint Mean Bias Penalty (Stabilizes p & T without drift)
        bias_p = torch.abs(torch.mean(p_pred) - torch.mean(target[:, self.p_idx, :, :]))
        bias_t = torch.abs(torch.mean(pred[:, 0, :, :]) - torch.mean(target[:, 0, :, :]))
        loss_joint_bias = bias_p + bias_t

        # Total Weighted Combination
        total_loss = (
            self.weight_mse * loss_mse
            + self.lambda_laplacian_p * loss_laplacian_p
            + self.weight_grad_state * loss_grad_state
            + self.lambda_asym_p * loss_asym_p
            + self.weight_state_eq * loss_state_eq
            + self.weight_q_log * loss_q_log
            + self.weight_joint_bias * loss_joint_bias
        )

        loss_metrics = {
            "loss_total": total_loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_laplacian_p": loss_laplacian_p.item(),
            "loss_grad_state": loss_grad_state.item(),
            "loss_asym_p": loss_asym_p.item(),
            "loss_state_eq": loss_state_eq.item(),
            "loss_q_log": loss_q_log.item(),
            "loss_joint_bias": loss_joint_bias.item(),
        }

        return total_loss, loss_metrics

# =============================================================================
# 5. TRAINING PIPELINE EXECUTION
# =============================================================================
def train_model(args):
    print(f"[TRAIN] Beginning training for {args.epochs} epochs...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Operating on compute device: {device}", flush=True)

    if args.zarr and os.path.exists(args.zarr):
        dataset = LogStateZarrDataset(zarr_path=args.zarr)
        num_nodes = dataset.num_nodes
    else:
        print(f"[WARNING] Zarr dataset path '{args.zarr}' not found. Falling back to synthetic dataset.")
        dataset = SyntheticAIDAStateDataset(num_samples=args.samples, num_nodes=args.num_nodes, num_levels=args.levels)
        num_nodes = args.num_nodes

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    edge_index = generate_or_load_edge_index(num_nodes=num_nodes, edge_file=args.edges).to(device)

    model = IcosahedralGNNSurrogate(
        in_vars=dataset.num_vars if hasattr(dataset, 'num_vars') else 7,
        hidden_dim=args.hidden_dim
    ).to(device)

    criterion = AIDASurrogateLoss(
        lambda_laplacian_p=args.lambda_laplacian_p,
        weight_grad_state=args.weight_grad_state,
        lambda_asym_p=args.lambda_asym_p,
        weight_state_eq=args.weight_state_eq,
        weight_q_log=args.weight_q_log,
        weight_joint_bias=args.weight_joint_bias,
        tau_min_p=args.tau_min_p
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"[TRAIN] Beginning training for {args.epochs} epochs...", flush=True)
    print(f"[TRAIN] Pressure Laplacian weight (lambda_laplacian_p): {args.lambda_laplacian_p}", flush=True)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {}  # Dynamically populated on first batch iteration

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch, edge_index)
            loss, metrics = criterion(pred, y_batch, edge_index)

            # Prevent NaN propagation in backward step
            if torch.isnan(loss):
                print(f"[WARNING] NaN loss detected in batch! Skipping step...")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Initialize keys dynamically on first batch
            if not epoch_losses:
                epoch_losses = {k: 0.0 for k in metrics.keys()}

            for k, v in metrics.items():
                epoch_losses[k] += v / len(dataloader)

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"Total: {epoch_losses['loss_total']:.4f} | "
                f"MSE: {epoch_losses['loss_mse']:.4f} | "
                f"Laplacian_P: {epoch_losses['loss_laplacian_p']:.5f} | "
                f"Grad_State: {epoch_losses['loss_grad_state']:.4f} | "
                f"Q_Log: {epoch_losses.get('loss_q_log', 0.0):.4f}"
            )

    # Save model dict along with 'stats' dict so cycling scripts pick it up automatically
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "stats": {
            "mu_ln_t": criterion.mu_ln_t,
            "std_ln_t": criterion.std_ln_t,
            "mu_ln_rho": criterion.mu_ln_rho,
            "std_ln_rho": criterion.std_ln_rho,
            "mu_ln_p": criterion.mu_ln_p,
            "std_ln_p": criterion.std_ln_p,
        }
    }, args.checkpoint)
    print(f"[TRAIN] Checkpoint successfully saved to '{args.checkpoint}'")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train AIDA GNN Surrogate Model")

    parser.add_argument("--zarr", type=str, default="../data/icosahedral_2023_logstate.zarr", help="Path to input Zarr dataset")
    parser.add_argument("--edges", type=str, default="../data/graph/icosahedral_edge_index_m4.pt", help="Path to precomputed edge tensor (.pt)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/aida_gnn_surrogate_logstate.pt", help="Checkpoint output path")

    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    parser.add_argument("--num-nodes", "--num_nodes", dest="num_nodes", type=int, default=2562)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=64)

    parser.add_argument(
        "--lambda-laplacian-p", "--lambda_laplacian_p",
        dest="lambda_laplacian_p", type=float, default=0.18,  # Tuned down from 0.30
        help="2nd-order graph Laplacian weight for pressure"
    )
    parser.add_argument(
        "--weight-grad-state", "--weight_grad_state",
        dest="weight_grad_state", type=float, default=0.25,   # Boosted from 0.20 to sharpen fronts
        help="State gradient matching weight"
    )
    parser.add_argument(
        "--weight-state-eq", "--weight_state_eq",
        dest="weight_state_eq", type=float, default=0.15,     # Increased from 0.10 for physical consistency
        help="Ideal gas residual weight"
    )
    parser.add_argument("--lambda-asym-p", "--lambda_asym_p", dest="lambda_asym_p", type=float, default=0.25, help="Asymmetric pressure penalty weight")
    parser.add_argument("--tau-min-p", "--tau_min_p", dest="tau_min_p", type=float, default=-0.2894, help="Low-pressure barrier threshold")

    parser.add_argument("--log-interval", "--log_interval", dest="log_interval", type=int, default=2, help="Logging epoch frequency")

    parser.add_argument(
        "--weight-q-log", "--weight_q_log",
        dest="weight_q_log", type=float, default=0.15,
        help="Log-scale loss weight for specific humidity (q)"
    )
    parser.add_argument(
        "--weight-joint-bias", "--weight_joint_bias",
        dest="weight_joint_bias", type=float, default=0.05,
        help="Joint p and T mean bias penalty weight"
    )

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
