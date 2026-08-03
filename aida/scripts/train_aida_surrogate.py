#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
AIDA GNN Surrogate Model Training Script for Icosahedral Atmospheric Grids.
Refactored to import modules from models/ directory.
"""

import argparse
import os
import sys

# Ensure parent directory is in Python path for 'models' package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from models import (
    LogStateZarrDataset,
    SyntheticAIDAStateDataset,
    generate_or_load_edge_index,
    IcosahedralGNNSurrogate,
    AIDASurrogateLoss,
)


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

    print(f"[TRAIN] Pressure Laplacian weight (lambda_laplacian_p): {args.lambda_laplacian_p}", flush=True)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {}  # Dynamic metrics accumulator

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch, edge_index)
            loss, metrics = criterion(pred, y_batch, edge_index)

            if torch.isnan(loss):
                print("[WARNING] NaN loss detected in batch! Skipping step...")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Dynamic initialization on first batch
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

    # Save checkpoint with normalization stats for cycling
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
        dest="lambda_laplacian_p", type=float, default=0.18,
        help="2nd-order graph Laplacian weight for pressure"
    )
    parser.add_argument(
        "--weight-grad-state", "--weight_grad_state",
        dest="weight_grad_state", type=float, default=0.20,
        help="State gradient matching weight"
    )
    parser.add_argument(
        "--weight-state-eq", "--weight_state_eq",
        dest="weight_state_eq", type=float, default=0.10,
        help="Ideal gas residual weight"
    )
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
    parser.add_argument("--lambda-asym-p", "--lambda_asym_p", dest="lambda_asym_p", type=float, default=0.25, help="Asymmetric pressure penalty weight")
    parser.add_argument("--tau-min-p", "--tau_min_p", dest="tau_min_p", type=float, default=-0.2894, help="Low-pressure barrier threshold")

    parser.add_argument("--log-interval", "--log_interval", dest="log_interval", type=int, default=2, help="Logging epoch frequency")

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
