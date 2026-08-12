#!/usr/bin/env python3
"""
train_aida_surrogate.py
-----------------------
AIDA GNN Surrogate Model Training Script for Icosahedral Atmospheric Grids.
Supports differentiable AMSU-A, IASI, HMS, ATMS, CrIS, and SEVIRI radiance loss
integration with gradient accumulation for memory optimization.
"""

import argparse
import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
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
from models.hms import DifferentiableHMSOperator
from models.atms import DifferentiableATMSOperator
from models.cris import DifferentiableCrISOperator
from models.seviri import DifferentiableSEVIRIOperator
from models.gsrasr import DifferentiableGSRASROperator


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found at: '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(filepath: str, model, optimizer, epoch: int, cfg: dict, criterion):
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
    amsua_op, amsua_obs_err,
    iasi_op, iasi_obs_err,
    hms_op, hms_obs_err,
    atms_op, atms_obs_err,
    cris_op, cris_obs_err,
    seviri_op, seviri_obs_err,
    gsrasr_op, gsrasr_obs_err,
    loss_cfg,
    accum_steps: int = 4
):
    model.train()
    epoch_losses = {}
    num_batches = len(dataloader)
    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(dataloader):

        if isinstance(batch_data, dict):
            x_batch = batch_data['background'].to(device)
            y_batch = batch_data['target'].to(device)
            obs_amsua_tb = batch_data.get('obs_amsua_tb', None)
            obs_amsua_mask = batch_data.get('obs_amsua_mask', None)
            obs_iasi_tb = batch_data.get('obs_iasi_tb', None)
            obs_iasi_mask = batch_data.get('obs_iasi_mask', None)
            obs_hms_tb = batch_data.get('obs_hms_tb', None)
            obs_hms_mask = batch_data.get('obs_hms_mask', None)
            obs_atms_tb = batch_data.get('obs_atms_tb', None)
            obs_atms_mask = batch_data.get('obs_atms_mask', None)
            obs_cris_tb = batch_data.get('obs_cris_tb', None)
            obs_cris_mask = batch_data.get('obs_cris_mask', None)
            obs_seviri_tb = batch_data.get('obs_seviri_tb', None)
            obs_seviri_mask = batch_data.get('obs_seviri_mask', None)
            obs_gsrasr_tb = batch_data.get('obs_gsrasr_tb', None)
            obs_gsrasr_mask = batch_data.get('obs_gsrasr_mask', None)
            obs_conv_val = batch_data.get('obs_conv_val', None)
            obs_conv_mask = batch_data.get('obs_conv_mask', None)
        else:
            x_batch = batch_data[0].to(device)
            y_batch = batch_data[1].to(device)
            obs_amsua_tb, obs_amsua_mask = None, None
            obs_iasi_tb, obs_iasi_mask = None, None
            obs_hms_tb, obs_hms_mask = None, None
            obs_atms_tb, obs_atms_mask = None, None
            obs_cris_tb, obs_cris_mask = None, None
            obs_seviri_tb, obs_seviri_mask = None, None
            obs_gsrasr_tb, obs_gsrasr_mask = None, None
            obs_conv_val, obs_conv_mask = None, None

        # GNN Forward Pass
        pred = model(x_batch, edge_index)

        # 1. Base Physical Loss
        loss, metrics = criterion(
            pred=pred,
            target=y_batch,
            edge_index=edge_index,
            graph_mesh_ops=graph_mesh_ops
        )
        total_loss = loss

        # Un-normalize physical profile fields
        std_t = getattr(criterion, "std_ln_t", 1.0)
        mu_t = getattr(criterion, "mu_ln_t", 0.0)
        std_p = getattr(criterion, "std_ln_p", 1.0)
        mu_p = getattr(criterion, "mu_ln_p", 0.0)

        ln_T_phys = pred[:, 0, :, :].permute(0, 2, 1) * std_t + mu_t
        ln_p_phys = pred[:, 6, :, :].permute(0, 2, 1) * std_p + mu_p

        t_k = torch.clamp(torch.exp(ln_T_phys), min=180.0, max=330.0)
        p_hpa = torch.clamp(torch.exp(ln_p_phys) / 100.0, min=0.01, max=1050.0)

        # 2. Conventional Observation Loss
        w_conv = loss_cfg.get("w_conv", 0.05)
        if obs_conv_val is not None:
            conv_val = obs_conv_val.to(device)
            if obs_conv_mask is not None:
                conv_m = obs_conv_mask.to(device)
                loss_conv = torch.sum(((pred - conv_val) ** 2) * conv_m) / (torch.sum(conv_m) + 1e-8)
            else:
                loss_conv = F.mse_loss(pred, conv_val)
        else:
            loss_conv = F.mse_loss(pred[:, [0, 1, 2, 4, 6], :, :], y_batch[:, [0, 1, 2, 4, 6], :, :])

        total_loss += (w_conv * loss_conv)
        metrics["loss_conv"] = loss_conv.item()

        # 3. AMSU-A Radiance Loss
        w_rad_amsua = loss_cfg.get("w_rad", loss_cfg.get("w_rad_amsua", 0.01))
        if obs_amsua_tb is not None:
            tb_sim = amsua_op(t_k, p_hpa)
            tb_obs = obs_amsua_tb.to(device)
            if tb_sim.shape[1] != 15 and tb_sim.shape[2] == 15:
                tb_sim = tb_sim.permute(0, 2, 1)
            if tb_obs.shape[1] != 15 and tb_obs.shape[2] == 15:
                tb_obs = tb_obs.permute(0, 2, 1)
            err = amsua_obs_err.view(1, 15, 1)
            innov = (tb_obs - tb_sim) / err
            if obs_amsua_mask is not None:
                m = obs_amsua_mask.to(device)
                if m.shape[1] != 15 and m.shape[2] == 15:
                    m = m.permute(0, 2, 1)
                loss_rad_amsua = torch.sum((innov ** 2) * m) / (15.0 * torch.sum(m) + 1e-8)
            else:
                loss_rad_amsua = torch.mean(innov ** 2) / 15.0
        else:
            loss_rad_amsua = torch.tensor(0.0, device=device)
        total_loss += (w_rad_amsua * loss_rad_amsua)
        metrics["loss_rad_amsua"] = loss_rad_amsua.item()

        # 4. IASI Radiance Loss
        w_rad_iasi = loss_cfg.get("w_rad_iasi", 0.01)
        if obs_iasi_tb is not None:
            p_pa = p_hpa.permute(0, 2, 1) * 100.0
            t_k_perm = t_k.permute(0, 2, 1)
            tb_sim = iasi_op(t_k_perm, p_pa)
            tb_obs = obs_iasi_tb.to(device)
            if tb_obs.shape[1] != 30 and tb_obs.shape[2] == 30:
                tb_obs = tb_obs.permute(0, 2, 1)
            if tb_sim.shape[1] != 30 and tb_sim.shape[2] == 30:
                tb_sim = tb_sim.permute(0, 2, 1)
            err = iasi_obs_err.view(1, 30, 1)
            innov = (tb_obs - tb_sim) / err
            if obs_iasi_mask is not None:
                m = obs_iasi_mask.to(device)
                if m.shape[1] != 30 and m.shape[2] == 30:
                    m = m.permute(0, 2, 1)
                loss_rad_iasi = torch.sum((innov ** 2) * m) / (30.0 * torch.sum(m) + 1e-8)
            else:
                loss_rad_iasi = torch.mean(innov ** 2) / 30.0
        else:
            loss_rad_iasi = torch.tensor(0.0, device=device)
        total_loss += (w_rad_iasi * loss_rad_iasi)
        metrics["loss_rad_iasi"] = loss_rad_iasi.item()

        # 5. HMS Radiance Loss
        w_rad_hms = loss_cfg.get("w_rad_hms", 0.01)
        if obs_hms_tb is not None:
            p_pa = p_hpa.permute(0, 2, 1) * 100.0
            t_k_perm = t_k.permute(0, 2, 1)
            tb_sim = hms_op(t_k_perm, p_pa)
            tb_obs = obs_hms_tb.to(device)
            if tb_obs.shape[1] != 12 and tb_obs.shape[2] == 12:
                tb_obs = tb_obs.permute(0, 2, 1)
            if tb_sim.shape[1] != 12 and tb_sim.shape[2] == 12:
                tb_sim = tb_sim.permute(0, 2, 1)
            err = hms_obs_err.view(1, 12, 1)
            innov = (tb_obs - tb_sim) / err
            if obs_hms_mask is not None:
                m = obs_hms_mask.to(device)
                if m.shape[1] != 12 and m.shape[2] == 12:
                    m = m.permute(0, 2, 1)
                loss_rad_hms = torch.sum((innov ** 2) * m) / (12.0 * torch.sum(m) + 1e-8)
            else:
                loss_rad_hms = torch.mean(innov ** 2) / 12.0
        else:
            loss_rad_hms = torch.tensor(0.0, device=device)
        total_loss += (w_rad_hms * loss_rad_hms)
        metrics["loss_rad_hms"] = loss_rad_hms.item()

        # 6. ATMS Radiance Loss
        w_rad_atms = loss_cfg.get("w_rad_atms", 0.01)
        if obs_atms_tb is not None:
            p_pa = p_hpa.permute(0, 2, 1) * 100.0
            t_k_perm = t_k.permute(0, 2, 1)
            tb_sim = atms_op(t_k_perm, p_pa)
            tb_obs = obs_atms_tb.to(device)
            if tb_obs.shape[1] != 22 and tb_obs.shape[2] == 22:
                tb_obs = tb_obs.permute(0, 2, 1)
            if tb_sim.shape[1] != 22 and tb_sim.shape[2] == 22:
                tb_sim = tb_sim.permute(0, 2, 1)
            err = atms_obs_err.view(1, 22, 1)
            innov = (tb_obs - tb_sim) / err
            if obs_atms_mask is not None:
                m = obs_atms_mask.to(device)
                if m.shape[1] != 22 and m.shape[2] == 22:
                    m = m.permute(0, 2, 1)
                loss_rad_atms = torch.sum((innov ** 2) * m) / (22.0 * torch.sum(m) + 1e-8)
            else:
                loss_rad_atms = torch.mean(innov ** 2) / 22.0
        else:
            loss_rad_atms = torch.tensor(0.0, device=device)
        total_loss += (w_rad_atms * loss_rad_atms)
        metrics["loss_rad_atms"] = loss_rad_atms.item()

        # 7. CrIS Radiance Loss
        w_rad_cris = loss_cfg.get("w_rad_cris", 0.01)
        if obs_cris_tb is not None:
            p_pa = p_hpa.permute(0, 2, 1) * 100.0
            t_k_perm = t_k.permute(0, 2, 1)
            tb_sim = cris_op(t_k_perm, p_pa)
            tb_obs = obs_cris_tb.to(device)
            if tb_obs.shape[1] != 30 and tb_obs.shape[2] == 30:
                tb_obs = tb_obs.permute(0, 2, 1)
            if tb_sim.shape[1] != 30 and tb_sim.shape[2] == 30:
                tb_sim = tb_sim.permute(0, 2, 1)
            err = cris_obs_err.view(1, 30, 1)
            innov = (tb_obs - tb_sim) / err
            if obs_cris_mask is not None:
                m = obs_cris_mask.to(device)
                if m.shape[1] != 30 and m.shape[2] == 30:
                    m = m.permute(0, 2, 1)
                loss_rad_cris = torch.sum((innov ** 2) * m) / (30.0 * torch.sum(m) + 1e-8)
            else:
                loss_rad_cris = torch.mean(innov ** 2) / 30.0
        else:
            loss_rad_cris = torch.tensor(0.0, device=device)
        total_loss += (w_rad_cris * loss_rad_cris)
        metrics["loss_rad_cris"] = loss_rad_cris.item()

        # 8. SEVIRI Radiance Loss
        w_rad_seviri = loss_cfg.get("w_rad_seviri", 0.01)
        if obs_seviri_tb is not None:
            p_pa = p_hpa.permute(0, 2, 1) * 100.0
            t_k_perm = t_k.permute(0, 2, 1)
            tb_sim = seviri_op(t_k_perm, p_pa)
            tb_obs = obs_seviri_tb.to(device)
            if tb_obs.shape[1] != 8 and tb_obs.shape[2] == 8:
                tb_obs = tb_obs.permute(0, 2, 1)
            if tb_sim.shape[1] != 8 and tb_sim.shape[2] == 8:
                tb_sim = tb_sim.permute(0, 2, 1)
            err = seviri_obs_err.view(1, 8, 1)
            innov = (tb_obs - tb_sim) / err
            if obs_seviri_mask is not None:
                m = obs_seviri_mask.to(device)
                if m.shape[1] != 8 and m.shape[2] == 8:
                    m = m.permute(0, 2, 1)
                loss_rad_seviri = torch.sum((innov ** 2) * m) / (8.0 * torch.sum(m) + 1e-8)
            else:
                loss_rad_seviri = torch.mean(innov ** 2) / 8.0
        else:
            loss_rad_seviri = torch.tensor(0.0, device=device)
        total_loss += (w_rad_seviri * loss_rad_seviri)
        metrics["loss_rad_seviri"] = loss_rad_seviri.item()

        # 8. Evaluate GSRASR Radiance Innovation Loss
        w_rad_gsrasr = loss_cfg.get("w_rad_gsrasr", 0.01)
        if obs_gsrasr_tb is not None:
            p_pa = p_hpa.permute(0, 2, 1) * 100.0
            t_k_perm = t_k.permute(0, 2, 1)
            tb_gsrasr_sim = gsrasr_op(t_k_perm, p_pa)  # [B, 10, N]

            tb_gsrasr_obs = obs_gsrasr_tb.to(device)
            if tb_gsrasr_obs.shape[1] != 10 and tb_gsrasr_obs.shape[2] == 10:
                tb_gsrasr_obs = tb_gsrasr_obs.permute(0, 2, 1)

            if tb_gsrasr_sim.shape[1] != 10 and tb_gsrasr_sim.shape[2] == 10:
                tb_gsrasr_sim = tb_gsrasr_sim.permute(0, 2, 1)

            err_gsrasr = gsrasr_obs_err.view(1, 10, 1)
            innov_gsrasr = (tb_gsrasr_obs - tb_gsrasr_sim) / err_gsrasr

            if obs_gsrasr_mask is not None:
                mask_gsrasr = obs_gsrasr_mask.to(device)
                if mask_gsrasr.shape[1] != 10 and mask_gsrasr.shape[2] == 10:
                    mask_gsrasr = mask_gsrasr.permute(0, 2, 1)
                loss_rad_gsrasr = torch.sum((innov_gsrasr ** 2) * mask_gsrasr) / (10.0 * torch.sum(mask_gsrasr) + 1e-8)
            else:
                loss_rad_gsrasr = torch.mean(innov_gsrasr ** 2) / 10.0
        else:
            loss_rad_gsrasr = torch.tensor(0.0, device=device)
    
        total_loss += (w_rad_gsrasr * loss_rad_gsrasr)
        metrics["loss_rad_gsrasr"] = loss_rad_gsrasr.item()

        # ------------------------------------------------------------------------------------------------------------
        metrics["loss_total"] = total_loss.item()

        if torch.isnan(total_loss):
            print("[WARNING] NaN loss detected in batch! Skipping step...", flush=True)
            continue

        # Scale loss for gradient accumulation to conserve memory
        loss_accum = total_loss / accum_steps
        loss_accum.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

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

    zarr_path = paths["zarr_path"]
    if zarr_path and os.path.exists(zarr_path):
        print(f"[TRAIN] Loading dataset from Zarr: '{zarr_path}'", flush=True)
        obs_dir = paths.get("obs_dir", None)
        if obs_dir and os.path.exists(obs_dir):
            print(f"[TRAIN] Loading dataset from Obs: '{obs_dir}'", flush=True)
            dataset = LogStateZarrDataset(zarr_path=zarr_path, obs_dir=obs_dir)
        else:
            dataset = LogStateZarrDataset(zarr_path=zarr_path)
        num_nodes = dataset.num_nodes
        lat_deg = torch.tensor(dataset.latitudes, dtype=torch.float32) if hasattr(dataset, "latitudes") else torch.linspace(-90, 90, num_nodes)
        lon_deg = torch.tensor(dataset.longitudes, dtype=torch.float32) if hasattr(dataset, "longitudes") else torch.linspace(-180, 180, num_nodes)
    else:
        dataset = SyntheticAIDAStateDataset(
            num_samples=mesh_cfg["samples"],
            num_nodes=mesh_cfg["num_nodes"],
            num_levels=mesh_cfg["num_levels"]
        )
        num_nodes = mesh_cfg["num_nodes"]
        lat_deg = torch.linspace(-90, 90, num_nodes)
        lon_deg = torch.linspace(-180, 180, num_nodes)

    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)

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

    num_levels = mesh_cfg.get("num_levels", 32)

    model = IcosahedralGNNSurrogate(
        in_vars=dataset.num_vars if hasattr(dataset, "num_vars") else 7,
        hidden_dim=model_cfg["hidden_dim"],
        num_levels=num_levels
    ).to(device)

    criterion = AIDASurrogateLoss(num_levels=num_levels, **loss_cfg).to(device)

    # Radiance Operators
    amsua_op = DifferentiableAMSUAOperator().to(device)
    amsua_obs_err = torch.tensor([
        2.5, 2.2, 1.2, 0.6, 0.3, 0.25, 0.25, 0.25,
        0.25, 0.35, 0.55, 0.8, 1.2, 1.8, 3.5
    ], dtype=torch.float32, device=device)

    iasi_op = DifferentiableIASIOperator(num_levels=num_levels).to(device)
    iasi_obs_err = iasi_op.obs_errors.to(device)

    hms_op = DifferentiableHMSOperator(num_levels=num_levels).to(device)
    hms_obs_err = hms_op.obs_errors.to(device)

    atms_op = DifferentiableATMSOperator(num_levels=num_levels).to(device)
    atms_obs_err = atms_op.obs_errors.to(device)

    cris_op = DifferentiableCrISOperator(num_levels=num_levels).to(device)
    cris_obs_err = cris_op.obs_errors.to(device)

    seviri_op = DifferentiableSEVIRIOperator(num_levels=num_levels).to(device)
    seviri_obs_err = seviri_op.obs_errors.to(device)

    gsrasr_op = DifferentiableGSRASROperator(num_levels=num_levels).to(device)
    gsrasr_obs_err = gsrasr_op.obs_errors.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4)
    )

    print(f"[TRAIN] AMSU-A Radiance Weight: {loss_cfg.get('w_rad_amsua', 0.01)}", flush=True)
    print(f"[TRAIN] IASI Radiance Weight  : {loss_cfg.get('w_rad_iasi', 0.01)}", flush=True)
    print(f"[TRAIN] HMS Radiance Weight   : {loss_cfg.get('w_rad_hms', 0.01)}", flush=True)
    print(f"[TRAIN] ATMS Radiance Weight  : {loss_cfg.get('w_rad_atms', 0.01)}", flush=True)
    print(f"[TRAIN] CrIS Radiance Weight  : {loss_cfg.get('w_rad_cris', 0.01)}", flush=True)
    print(f"[TRAIN] SEVIRI Radiance Weight: {loss_cfg.get('w_rad_seviri', 0.01)}", flush=True)
    print(f"[TRAIN] GSRASR Radiance Weight: {loss_cfg.get('w_rad_gsrasr', 0.01)}", flush=True)

    checkpoint_path = paths["checkpoint_path"]
    save_interval = train_cfg.get("save_interval", 5)
    epochs = train_cfg["epochs"]
    log_interval = train_cfg["log_interval"]
    accum_steps = train_cfg.get("accum_steps", 4)

    for epoch in range(1, epochs + 1):
        epoch_losses = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            edge_index=edge_index,
            graph_mesh_ops=graph_mesh_ops,
            amsua_op=amsua_op, amsua_obs_err=amsua_obs_err,
            iasi_op=iasi_op, iasi_obs_err=iasi_obs_err,
            hms_op=hms_op, hms_obs_err=hms_obs_err,
            atms_op=atms_op, atms_obs_err=atms_obs_err,
            cris_op=cris_op, cris_obs_err=cris_obs_err,
            seviri_op=seviri_op, seviri_obs_err=seviri_obs_err,
            gsrasr_op=gsrasr_op, gsrasr_obs_err=gsrasr_obs_err,
            loss_cfg=loss_cfg,
            accum_steps=accum_steps
        )

        if epoch % log_interval == 0 or epoch == epochs:
            print(
                f"\n" + "=" * 110 + "\n"
                f" EPOCH {epoch:03d}/{epochs:03d} TRAINING LOSS BREAKDOWN\n"
                f" " + "-" * 108 + "\n"
                f"  TOTAL LOSS    : {epoch_losses.get('loss_total', 0.0):12.5e} | STATE MSE     : {epoch_losses.get('loss_mse', 0.0):12.5e}\n"
                f"  CONV OBS LOSS : {epoch_losses.get('loss_conv', 0.0):12.5e} | LAPLACIAN P   : {epoch_losses.get('loss_laplacian_p', 0.0):12.5e}\n"
                f"  AMSU-A RAD    : {epoch_losses.get('loss_rad_amsua', 0.0):12.5e} | IASI RAD      : {epoch_losses.get('loss_rad_iasi', 0.0):12.5e}\n"
                f"  HMS RAD       : {epoch_losses.get('loss_rad_hms', 0.0):12.5e} | ATMS RAD      : {epoch_losses.get('loss_rad_atms', 0.0):12.5e}\n"
                f"  CrIS RAD      : {epoch_losses.get('loss_rad_cris', 0.0):12.5e} | SEVIRI RAD    : {epoch_losses.get('loss_rad_seviri', 0.0):12.5e}\n"
                f"  GSRASR RAD    : {epoch_losses.get('loss_rad_gsrasr', 0.0):12.5e}\n"
                f"  DYNAMICS LOSS : {epoch_losses.get('loss_dynamics_total', 0.0):12.5e} | JOINT BIAS    : {epoch_losses.get('loss_joint_bias', 0.0):12.5e}\n"
                f"=" * 110,
                flush=True
            )

        if epoch % save_interval == 0 or epoch == epochs:
            base_name, ext = os.path.splitext(checkpoint_path)
            epoch_ckpt_path = f"{base_name}_epoch_{epoch:03d}{ext}"
            save_checkpoint(epoch_ckpt_path, model, optimizer, epoch, cfg, criterion)
            save_checkpoint(checkpoint_path, model, optimizer, epoch, cfg, criterion)


def main():
    parser = argparse.ArgumentParser(description="Train AIDA GNN Surrogate Model via YAML Config")
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_model(cfg)


if __name__ == "__main__":
    main()
