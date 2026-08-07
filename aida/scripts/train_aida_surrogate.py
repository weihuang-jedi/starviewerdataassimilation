#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
AIDA GNN Surrogate Model Training Script for Icosahedral Atmospheric Grids.
Reads runtime parameters, loss weights, and file paths from a YAML configuration file.
Supports differentiable AMSU-A and IASI radiance loss integration, dynamic mesh operators,
and periodic checkpoint saving every N epochs.
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
from models.iasi import DifferentiableIASIOperator


def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found at: '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(filepath: str, model, optimizer, epoch: int, cfg: dict, criterion):
    """Utility to serialize model checkpoint to disk."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    stats_dict = {}
    for attr in ["mu_ln_t", "std_ln_t", "mu_ln_rho", "std_ln_rho", "mu_ln_p", "std_ln_p"]:
        if hasattr(criterion, attr):
            stats_dict[attr] = getattr(criterion, attr)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "stats": stats_dict
    }, filepath)
    print(f"[TRAIN] Checkpoint successfully saved to '{filepath}'", flush=True)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    edge_index,
    graph_mesh_ops,
    amsua_op,
    amsua_obs_err,
    iasi_op,
    iasi_obs_err,
    loss_cfg
):
    """Executes a single training epoch across the dataset."""
    model.train()
    epoch_losses = {}
    num_batches = len(dataloader)

    for batch_data in dataloader:
        optimizer.zero_grad()

        # Handle different dataset batch return shapes
        if isinstance(batch_data, dict):
            x_batch = batch_data['background'].to(device)
            y_batch = batch_data['target'].to(device)
            obs_amsua_tb = batch_data.get('obs_amsua_tb', None)
            obs_amsua_mask = batch_data.get('obs_amsua_mask', None)
            obs_iasi_tb = batch_data.get('obs_iasi_tb', None)
            obs_iasi_mask = batch_data.get('obs_iasi_mask', None)
        elif isinstance(batch_data, (tuple, list)):
            x_batch = batch_data[0].to(device)
            y_batch = batch_data[1].to(device)
            obs_amsua_tb = batch_data[2].to(device) if len(batch_data) >= 3 else None
            obs_amsua_mask = batch_data[3].to(device) if len(batch_data) >= 4 else None
            obs_iasi_tb = batch_data[4].to(device) if len(batch_data) >= 5 else None
            obs_iasi_mask = batch_data[5].to(device) if len(batch_data) >= 6 else None
        else:
            x_batch = batch_data.to(device)
            y_batch = x_batch
            obs_amsua_tb, obs_amsua_mask = None, None
            obs_iasi_tb, obs_iasi_mask = None, None

        # GNN Forward Pass
        pred = model(x_batch, edge_index)

        # 1. Base Multi-Component Physical Loss Calculation
        loss, metrics = criterion(
            pred=pred,
            target=y_batch,
            edge_index=edge_index,
            graph_mesh_ops=graph_mesh_ops
        )

        # Permute log-state predictions for radiance evaluation: [B, V=7, L=32, N] -> [B, N, L=32]
        std_t = getattr(criterion, "std_ln_t", 1.0)
        mu_t = getattr(criterion, "mu_ln_t", 0.0)
        std_p = getattr(criterion, "std_ln_p", 1.0)
        mu_p = getattr(criterion, "mu_ln_p", 0.0)

        ln_T_phys = pred[:, 0, :, :].permute(0, 2, 1) * std_t + mu_t
        ln_p_phys = pred[:, 6, :, :].permute(0, 2, 1) * std_p + mu_p

        t_k = torch.clamp(torch.exp(ln_T_phys), min=180.0, max=330.0)
        p_hpa = torch.clamp(torch.exp(ln_p_phys) / 100.0, min=0.01, max=1050.0)

        # ---------------------------------------------------------------------
        # 2. Differentiable AMSU-A Radiance Innovation Loss
        # ---------------------------------------------------------------------
        w_rad_amsua = loss_cfg.get("w_rad", loss_cfg.get("w_rad_amsua", 0.01))
        if obs_amsua_tb is not None:
            tb_amsua_sim = amsua_op(t_k, p_hpa)  # Output: [B, N=2562, Ch=15] or [B, Ch=15, N=2562]

            # Ensure both simulated and observed match shape: [B, Channels=15, Nodes=2562]
            tb_amsua_obs = obs_amsua_tb.to(device)
            if tb_amsua_sim.shape[1] != 15 and tb_amsua_sim.shape[2] == 15:
                tb_amsua_sim = tb_amsua_sim.permute(0, 2, 1)  # -> [B, 15, 2562]

            if tb_amsua_obs.shape[1] != 15 and tb_amsua_obs.shape[2] == 15:
                tb_amsua_obs = tb_amsua_obs.permute(0, 2, 1)  # -> [B, 15, 2562]

            # Reshape amsua_obs_err [15] -> [1, 15, 1] for correct broadcasting across nodes
            err_amsua = amsua_obs_err.view(1, 15, 1)

            innov_amsua = (tb_amsua_obs - tb_amsua_sim) / err_amsua  # [B, 15, 2562]

            if obs_amsua_mask is not None:
                mask_amsua = obs_amsua_mask.to(device)
                if mask_amsua.shape[1] != 15 and mask_amsua.shape[2] == 15:
                    mask_amsua = mask_amsua.permute(0, 2, 1)
                loss_rad_amsua = torch.sum((innov_amsua ** 2) * mask_amsua) / (15.0 * torch.sum(mask_amsua) + 1e-8)
            else:
                loss_rad_amsua = torch.mean(innov_amsua ** 2) / 15.0
        else:
            loss_rad_amsua = torch.tensor(0.0, device=device)

        # ---------------------------------------------------------------------
        # 3. Differentiable IASI Radiance Innovation Loss
        # ---------------------------------------------------------------------
        w_rad_iasi = loss_cfg.get("w_rad_iasi", 0.01)
        if obs_iasi_tb is not None:
            p_pa = p_hpa.permute(0, 2, 1) * 100.0
            t_k_perm = t_k.permute(0, 2, 1)
            tb_iasi_sim = iasi_op(t_k_perm, p_pa)  # [B, 30, N=2562]

            tb_iasi_obs = obs_iasi_tb.to(device)
            if tb_iasi_obs.shape[1] != 30 and tb_iasi_obs.shape[2] == 30:
                tb_iasi_obs = tb_iasi_obs.permute(0, 2, 1)

            if tb_iasi_sim.shape[1] != 30 and tb_iasi_sim.shape[2] == 30:
                tb_iasi_sim = tb_iasi_sim.permute(0, 2, 1)

            err_iasi = iasi_obs_err.view(1, 30, 1)
            innov_iasi = (tb_iasi_obs - tb_iasi_sim) / err_iasi

            if obs_iasi_mask is not None:
                mask_iasi = obs_iasi_mask.to(device)
                if mask_iasi.shape[1] != 30 and mask_iasi.shape[2] == 30:
                    mask_iasi = mask_iasi.permute(0, 2, 1)
                loss_rad_iasi = torch.sum((innov_iasi ** 2) * mask_iasi) / (30.0 * torch.sum(mask_iasi) + 1e-8)
            else:
                loss_rad_iasi = torch.mean(innov_iasi ** 2) / 30.0
        else:
            loss_rad_iasi = torch.tensor(0.0, device=device)

        # 4. Total Combined Objective
        total_loss = loss + (w_rad_amsua * loss_rad_amsua) + (w_rad_iasi * loss_rad_iasi)

        metrics["loss_total"] = total_loss.item()
        metrics["loss_rad_amsua"] = loss_rad_amsua.item()
        metrics["loss_rad_iasi"] = loss_rad_iasi.item()

        if torch.isnan(total_loss):
            print("[WARNING] NaN loss detected in batch! Skipping step...", flush=True)
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if not epoch_losses:
            epoch_losses = {k: 0.0 for k in metrics.keys()}

        for k, v in metrics.items():
            epoch_losses[k] += v / num_batches

    return epoch_losses


def train_model(cfg: dict):
    paths = cfg["paths"]
    mesh_cfg = cfg["mesh"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss_weights"]

    print(f"[TRAIN] Beginning training for {train_cfg['epochs']} epochs...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Operating on compute device: {device}", flush=True)

    # 1. Dataset Initialization
    zarr_path = paths["zarr_path"]
    if zarr_path and os.path.exists(zarr_path):
        print(f"[TRAIN] Loading dataset from Zarr: '{zarr_path}'", flush=True)
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

    # 2. Graph Topology & Differential Operators
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

    # 3. Model & Operators Setup
    num_levels = mesh_cfg.get("num_levels", 32)

    # 4. Model, Loss, & Operators Setup
    model = IcosahedralGNNSurrogate(
        in_vars=dataset.num_vars if hasattr(dataset, "num_vars") else 7,
        hidden_dim=model_cfg["hidden_dim"],
        num_levels=num_levels
    ).to(device)

    criterion = AIDASurrogateLoss(
        num_levels=num_levels,
        **loss_cfg
    ).to(device)

    # Radiance Forward Operators
    amsua_op = DifferentiableAMSUAOperator().to(device)
    amsua_obs_err = torch.tensor([
        2.5, 2.2, 1.2, 0.6, 0.3, 0.25, 0.25, 0.25,
        0.25, 0.35, 0.55, 0.8, 1.2, 1.8, 3.5
    ], dtype=torch.float32, device=device)

    iasi_op = DifferentiableIASIOperator(num_levels=mesh_cfg.get("num_levels", 32)).to(device)
    iasi_obs_err = iasi_op.obs_errors.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4)
    )

    print(f"[TRAIN] AMSU-A Radiance Weight: {loss_cfg.get('w_rad', loss_cfg.get('w_rad_amsua', 0.01))}", flush=True)
    print(f"[TRAIN] IASI Radiance Weight: {loss_cfg.get('w_rad_iasi', 0.01)}", flush=True)
    print(f"[TRAIN] Dynamics Weight (lambda_dyn): {loss_cfg['lambda_dyn']}", flush=True)

    checkpoint_path = paths["checkpoint_path"]
    save_interval = train_cfg.get("save_interval", 5)
    print(f"[TRAIN] Checkpoints saved every {save_interval} epochs to: '{checkpoint_path}'", flush=True)

    # 4. Main Epoch Loop
    epochs = train_cfg["epochs"]
    log_interval = train_cfg["log_interval"]

    for epoch in range(1, epochs + 1):
        epoch_losses = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            edge_index=edge_index,
            graph_mesh_ops=graph_mesh_ops,
            amsua_op=amsua_op,
            amsua_obs_err=amsua_obs_err,
            iasi_op=iasi_op,
            iasi_obs_err=iasi_obs_err,
            loss_cfg=loss_cfg
        )

        # Logging
        if epoch % log_interval == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Total: {epoch_losses.get('loss_total', 0.0):.4f} | "
                f"MSE: {epoch_losses.get('loss_mse', 0.0):.4f} | "
                f"AMSUA_Rad: {epoch_losses.get('loss_rad_amsua', 0.0):.4f} | "
                f"IASI_Rad: {epoch_losses.get('loss_rad_iasi', 0.0):.4f} | "
                f"Laplacian_P: {epoch_losses.get('loss_laplacian_p', 0.0):.5f} | "
                f"Dyn_Total: {epoch_losses.get('loss_dynamics_total', 0.0):.5f}",
                flush=True
            )

        # Periodic Checkpointing
        if epoch % save_interval == 0 or epoch == epochs:
            base_name, ext = os.path.splitext(checkpoint_path)
            epoch_ckpt_path = f"{base_name}_epoch_{epoch:03d}{ext}"
            save_checkpoint(epoch_ckpt_path, model, optimizer, epoch, cfg, criterion)
            save_checkpoint(checkpoint_path, model, optimizer, epoch, cfg, criterion)


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
