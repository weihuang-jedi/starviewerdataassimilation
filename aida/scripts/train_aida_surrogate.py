#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
AIDA GNN Surrogate Model Training Script for Icosahedral Atmospheric Grids.
Reads runtime parameters, loss weights, and file paths from a YAML configuration file.
"""

import argparse
import os
import sys
import yaml

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
    M4MeshOperators,
    build_icosahedral_differential_operators,
)
from models.amsua import DifferentiableAMSUAOperator


def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found at: '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(cfg: dict):
    # Extract nested configuration sections
    paths = cfg["paths"]
    mesh_cfg = cfg["mesh"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss_weights"]

    print(f"[TRAIN] Beginning training for {train_cfg['epochs']} epochs...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Operating on compute device: {device}", flush=True)

    # 1. Dataset Loading
    zarr_path = paths["zarr_path"]
    if zarr_path and os.path.exists(zarr_path):
        print(f"[TRAIN] Loading real dataset from Zarr: '{zarr_path}'", flush=True)
        dataset = LogStateZarrDataset(zarr_path=zarr_path)
        num_nodes = dataset.num_nodes

        if hasattr(dataset, "latitudes") and hasattr(dataset, "longitudes"):
            lat_deg = torch.tensor(dataset.latitudes, dtype=torch.float32)
            lon_deg = torch.tensor(dataset.longitudes, dtype=torch.float32)
        else:
            lat_deg = torch.linspace(-90, 90, num_nodes)
            lon_deg = torch.linspace(-180, 180, num_nodes)
    else:
        print(f"[WARNING] Zarr path '{zarr_path}' not found. Falling back to synthetic dataset.", flush=True)
        dataset = SyntheticAIDAStateDataset(
            num_samples=mesh_cfg["samples"],
            num_nodes=mesh_cfg["num_nodes"],
            num_levels=mesh_cfg["num_levels"]
        )
        num_nodes = mesh_cfg["num_nodes"]
        lat_deg = torch.linspace(-90, 90, num_nodes)
        lon_deg = torch.linspace(-180, 180, num_nodes)

    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)

    # 2. Graph Edges & Sparse Differential Operators Initialization
    edge_index = generate_or_load_edge_index(
        num_nodes=num_nodes, 
        edge_file=paths["edges_path"]
    ).to(device)

    print("[TRAIN] Pre-computing M4 mesh sparse differential operators (Grad/Div)...", flush=True)
    Gx_sparse, Gy_sparse = build_icosahedral_differential_operators(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        edge_index=edge_index.cpu()
    )
    graph_mesh_ops = M4MeshOperators(
        Gx_sparse=Gx_sparse,
        Gy_sparse=Gy_sparse,
        lat_deg=lat_deg
    ).to(device)
    print("[TRAIN] Sparse differential operators successfully initialized on GPU.", flush=True)

    # 3. Model & Loss Instantiation
    model = IcosahedralGNNSurrogate(
        in_vars=dataset.num_vars if hasattr(dataset, "num_vars") else 7,
        hidden_dim=model_cfg["hidden_dim"]
    ).to(device)

    # Multi-component loss function
    criterion = AIDASurrogateLoss(
        lambda_laplacian_p=loss_cfg["lambda_laplacian_p"],
        weight_grad_state=loss_cfg["weight_grad_state"],
        lambda_asym_p=loss_cfg["lambda_asym_p"],
        weight_state_eq=loss_cfg["weight_state_eq"],
        weight_q_log=loss_cfg["weight_q_log"],
        weight_joint_bias=loss_cfg["weight_joint_bias"],
        lambda_dyn=loss_cfg["lambda_dyn"],
        tau_min_p=loss_cfg["tau_min_p"]
    ).to(device)

    # AMSU-A Radiance Forward Operator
    amsua_op = DifferentiableAMSUAOperator().to(device)
    amsua_obs_err = torch.tensor([
        2.5, 2.2, 1.2, 0.6, 0.3, 0.25, 0.25, 0.25,
        0.25, 0.35, 0.55, 0.8, 1.2, 1.8, 3.5
    ], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg["lr"], 
        weight_decay=train_cfg.get("weight_decay", 1e-4)
    )

    print(f"[TRAIN] AMSU-A Radiance Weight (w_rad): {loss_cfg['w_rad']}", flush=True)
    print(f"[TRAIN] Dynamics Weight (lambda_dyn): {loss_cfg['lambda_dyn']}", flush=True)
    print(f"[TRAIN] Pressure Laplacian Weight (lambda_laplacian_p): {loss_cfg['lambda_laplacian_p']}", flush=True)
    print(f"[TRAIN] Asymmetric Barrier Weight (lambda_asym_p): {loss_cfg['lambda_asym_p']}", flush=True)
    print(f"[TRAIN] Joint Bias Penalty Weight (weight_joint_bias): {loss_cfg['weight_joint_bias']}", flush=True)

    checkpoint_path = paths["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    # 4. Training Loop
    epochs = train_cfg["epochs"]
    log_interval = train_cfg["log_interval"]

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {}

        for batch_data in dataloader:
            if isinstance(batch_data, (tuple, list)):
                x_batch = batch_data[0].to(device)
                y_batch = batch_data[1].to(device)
                if len(batch_data) >= 4:
                    y_amsua = batch_data[2].to(device)
                    amsua_mask = batch_data[3].to(device)
                else:
                    batch_size = x_batch.shape[0]
                    y_amsua = torch.full((batch_size, num_nodes, 15), 240.0, device=device)
                    amsua_mask = torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)
            else:
                x_batch = batch_data.to(device)
                y_batch = x_batch
                batch_size = x_batch.shape[0]
                y_amsua = torch.full((batch_size, num_nodes, 15), 240.0, device=device)
                amsua_mask = torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)

            optimizer.zero_grad()
            pred = model(x_batch, edge_index)

            # Base physical loss evaluation
            loss, metrics = criterion(
                pred=pred,
                target=y_batch,
                edge_index=edge_index,
                graph_mesh_ops=graph_mesh_ops
            )

            # AMSU-A Radiance innovation loss
            # Slicing along variable dimension: Channel 0 = ln_T, Channel 6 = ln_p
            ln_T_pred = pred[:, 0, :, :].permute(0, 2, 1)  # -> [B, N_nodes, N_levels]
            ln_p_pred = pred[:, 6, :, :].permute(0, 2, 1)  # -> [B, N_nodes, N_levels]

            t_k = torch.exp(ln_T_pred)            # Kelvin
            p_hpa = torch.exp(ln_p_pred) / 100.0  # hPa
            tb_sim = amsua_op(t_k, p_hpa)          # [B, N_nodes, 15]

            innov = (y_amsua - tb_sim) / amsua_obs_err
            if amsua_mask is not None:
                mask_exp = amsua_mask.unsqueeze(-1).expand_as(innov)
                loss_rad = torch.sum((innov ** 2) * mask_exp) / (torch.sum(mask_exp) + 1e-8)
            else:
                loss_rad = torch.mean(innov ** 2)

            total_loss = loss + loss_cfg["w_rad"] * loss_rad
            metrics["loss_total"] = total_loss.item()
            metrics["loss_rad"] = loss_rad.item()

            if torch.isnan(total_loss):
                print("[WARNING] NaN loss detected in batch! Skipping step...", flush=True)
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if not epoch_losses:
                epoch_losses = {k: 0.0 for k in metrics.keys()}

            for k, v in metrics.items():
                epoch_losses[k] += v / len(dataloader)

        # Logging
        if epoch % log_interval == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Total: {epoch_losses['loss_total']:.4f} | "
                f"MSE: {epoch_losses['loss_mse']:.4f} | "
                f"Rad_Loss: {epoch_losses.get('loss_rad', 0.0):.4f} | "
                f"Laplacian_P: {epoch_losses['loss_laplacian_p']:.5f} | "
                f"Asym_P: {epoch_losses.get('loss_asym_p', 0.0):.5f} | "
                f"Joint_Bias: {epoch_losses.get('loss_joint_bias', 0.0):.5f} | "
                f"Dyn_Total: {epoch_losses.get('loss_dynamics_total', 0.0):.5f} | "
                f"Geo_Loss: {epoch_losses.get('loss_geostrophic', 0.0):.5f} | "
                f"Trop_Div: {epoch_losses.get('loss_tropical_div', 0.0):.5f} | "
                f"Q_Log: {epoch_losses.get('loss_q_log', 0.0):.4f}",
                flush=True
            )

    # 5. Checkpoint Saving
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "stats": {
            "mu_ln_t": criterion.mu_ln_t,
            "std_ln_t": criterion.std_ln_t,
            "mu_ln_rho": criterion.mu_ln_rho,
            "std_ln_rho": criterion.std_ln_rho,
            "mu_ln_p": criterion.mu_ln_p,
            "std_ln_p": criterion.std_ln_p,
        }
    }, checkpoint_path)
    print(f"[TRAIN] Checkpoint successfully saved to '{checkpoint_path}'", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train AIDA GNN Surrogate Model via YAML Config")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="configs/config.yaml", 
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_model(cfg)


if __name__ == "__main__":
    main()
