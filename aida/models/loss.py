#!/usr/bin/env python3
"""
models/loss.py
--------------
Composite Physical Balance, Mesh Laplacian, Thermal Drift, Boundary Layer, 
and Standardized Loss Engine for AIDA GNN Surrogate Model.
Enforces PBL height weighting, Monin-Obukhov drag penalties, and lapse rate bounds.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class M4MeshOperators(nn.Module):
    """Sparse differential operators (Gradient and Divergence) over icosahedral graph nodes."""
    def __init__(self, Gx_sparse: torch.Tensor, Gy_sparse: torch.Tensor, lat_deg: torch.Tensor):
        super().__init__()
        self.register_buffer("Gx_sparse", Gx_sparse)
        self.register_buffer("Gy_sparse", Gy_sparse)
        self.register_buffer("lat_deg", lat_deg)

    def forward(self, scalar_field: torch.Tensor):
        scalar_field = scalar_field.contiguous()
        orig_shape = scalar_field.shape

        if scalar_field.dim() == 3:
            B, L, N = orig_shape
            flat_field = scalar_field.reshape(B * L, N).t()
        else:
            B, N = orig_shape
            L = 1
            flat_field = scalar_field.t()

        # -----------------------------------------------------------------
        # FORCE FLOAT32 FOR BOTH FORWARD AND BACKWARD AUTOGRAD GRAPH
        # Do NOT cast back to orig_dtype (bfloat16/float16)!
        # -----------------------------------------------------------------
        flat_field_f32 = flat_field.float()

        df_dx_flat = torch.sparse.mm(self.Gx_sparse, flat_field_f32).t()
        df_dy_flat = torch.sparse.mm(self.Gy_sparse, flat_field_f32).t()

        if L > 1:
            df_dx = df_dx_flat.reshape(B, L, N)
            df_dy = df_dy_flat.reshape(B, L, N)
        else:
            df_dx = df_dx_flat.reshape(B, N)
            df_dy = df_dy_flat.reshape(B, N)

        return df_dx, df_dy


def build_icosahedral_differential_operators(lat_deg: torch.Tensor, lon_deg: torch.Tensor, edge_index: torch.Tensor):
    """Builds sparse CSR gradient operators (Gx, Gy) for icosahedral graph nodes."""
    N = len(lat_deg)
    src_nodes, dst_nodes = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()

    rad = np.pi / 180.0
    R_earth = 6371000.0

    lats_rad = lat_deg.cpu().numpy() * rad
    lons_rad = lon_deg.cpu().numpy() * rad

    dlat = lats_rad[dst_nodes] - lats_rad[src_nodes]
    dlon = lons_rad[dst_nodes] - lons_rad[src_nodes]
    dlon = np.where(dlon > np.pi, dlon - 2 * np.pi, dlon)
    dlon = np.where(dlon < -np.pi, dlon + 2 * np.pi, dlon)

    dx = R_earth * np.cos(0.5 * (lats_rad[src_nodes] + lats_rad[dst_nodes])) * dlon
    dy = R_earth * dlat

    dist_sq = dx**2 + dy**2 + 1e-6
    weights_x = dx / dist_sq
    weights_y = dy / dist_sq

    indices = torch.from_numpy(np.vstack([dst_nodes, src_nodes])).long()
    values_x = torch.from_numpy(weights_x).float()
    values_y = torch.from_numpy(weights_y).float()

    Gx_sparse = torch.sparse_coo_tensor(indices, values_x, size=(N, N)).to_sparse_csr()
    Gy_sparse = torch.sparse_coo_tensor(indices, values_y, size=(N, N)).to_sparse_csr()

    return Gx_sparse, Gy_sparse


def generate_or_load_edge_index(num_nodes: int, edge_file: str = None) -> torch.Tensor:
    if edge_file and os.path.exists(edge_file):
        print(f"[GRAPH] Loading precomputed edge topology from '{edge_file}'...", flush=True)
        return torch.load(edge_file)

    print(f"[GRAPH] Generating synthetic icosahedral edges for {num_nodes} nodes...", flush=True)
    edges_src, edges_dst = [], []
    for i in range(num_nodes):
        neighbors = [(i + j) % num_nodes for j in range(1, 7)]
        for n in neighbors:
            edges_src.append(i)
            edges_dst.append(n)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    if edge_file:
        os.makedirs(os.path.dirname(edge_file) or ".", exist_ok=True)
        torch.save(edge_index, edge_file)
    return edge_index


class AIDASurrogateLoss(nn.Module):
    """Standardized Physical Balance, PBL Height-Weighted, and Surface Drag Loss Engine."""
    def __init__(
        self,
        w_mse: float = 1.0,
        w_conv: float = 0.05,
        lambda_dyn: float = 0.0001,
        lambda_laplacian_p: float = 0.01,
        lambda_asym_q: float = 0.05,
        lambda_thermal: float = 0.1,
        lambda_pbl_drag: float = 0.05,
        lambda_lapse_rate: float = 0.02,
        num_levels: int = 32,
        **kwargs
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_conv = w_conv
        self.lambda_dyn = lambda_dyn
        self.lambda_laplacian_p = lambda_laplacian_p
        self.lambda_asym_q = lambda_asym_q
        self.lambda_thermal = lambda_thermal
        self.lambda_pbl_drag = lambda_pbl_drag
        self.lambda_lapse_rate = lambda_lapse_rate
        self.num_levels = num_levels

        # Order: [ln_t, u, v, w, q, ln_rho, ln_p]
        self.register_buffer("var_means", torch.tensor([5.50, 0.00, 0.00, 0.00, 0.005, -0.20, 10.50], dtype=torch.float32).view(1, 7, 1, 1))
        self.register_buffer("var_stds",  torch.tensor([0.15, 12.5, 12.5, 0.80, 0.005,  0.80,  1.20], dtype=torch.float32).view(1, 7, 1, 1))

        self.register_buffer("mu_ln_t", torch.tensor(5.50, dtype=torch.float32))
        self.register_buffer("std_ln_t", torch.tensor(0.15, dtype=torch.float32))
        self.register_buffer("mu_ln_rho", torch.tensor(-0.20, dtype=torch.float32))
        self.register_buffer("std_ln_rho", torch.tensor(0.80, dtype=torch.float32))
        self.register_buffer("mu_ln_p", torch.tensor(10.50, dtype=torch.float32))
        self.register_buffer("std_ln_p", torch.tensor(1.20, dtype=torch.float32))
        self.register_buffer("mu_q", torch.tensor(0.005, dtype=torch.float32))
        self.register_buffer("std_q", torch.tensor(0.005, dtype=torch.float32))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor,
        graph_mesh_ops: nn.Module = None,
        valid_mask: torch.Tensor = None,
        h_3d: torch.Tensor = None,
        static_topo: torch.Tensor = None
    ):
        metrics = {}

        pred_clean = torch.clamp(pred, min=-10.0, max=10.0)
        target_clean = torch.clamp(target, min=-10.0, max=10.0)

        pred_norm = (pred_clean - self.var_means) / self.var_stds
        target_norm = (target_clean - self.var_means) / self.var_stds

        # -----------------------------------------------------------------
        # 1. Exponential Planetary Boundary Layer (PBL) Loss Weighting
        # Uses real 3D height h_3d (meters) to weight surface (z <= 2000m)
        # -----------------------------------------------------------------
        if h_3d is not None:
            if h_3d.dim() == 2:
                h_3d_4d = h_3d.unsqueeze(0).unsqueeze(1)  # [1, 1, 32, Nodes]
            elif h_3d.dim() == 3:
                h_3d_4d = h_3d.unsqueeze(1)               # [B, 1, 32, Nodes]
            else:
                h_3d_4d = h_3d

            pbl_weight = 1.0 + 3.0 * torch.exp(-h_3d_4d.to(pred.device) / 1500.0)
        else:
            pbl_weight = 1.0

        diff_sq = (pred_norm - target_norm) ** 2
        weighted_diff_sq = diff_sq * pbl_weight

        if valid_mask is not None:
            mask_7d = valid_mask.unsqueeze(1).expand_as(pred_norm)
            loss_mse = torch.sum(weighted_diff_sq * mask_7d) / (torch.sum(mask_7d) * 7.0 + 1e-8)
        else:
            loss_mse = torch.mean(weighted_diff_sq)

        loss_mse = torch.nan_to_num(loss_mse, nan=0.0)
        metrics["loss_mse"] = loss_mse.item()
        total_loss = self.w_mse * loss_mse

        # 2. Asymmetric Physical Moisture Barrier Loss (q_phys >= 0 kg/kg)
        q_phys = pred_clean[:, 4, :, :] * self.std_q + self.mu_q
        q_neg_penalty = torch.relu(-q_phys + 1e-7) ** 2
        loss_asym_q = torch.mean(q_neg_penalty) * 1000.0
        loss_asym_q = torch.nan_to_num(loss_asym_q, nan=0.0)
        metrics["loss_asym_q"] = loss_asym_q.item()
        total_loss += (self.lambda_asym_q * loss_asym_q)

        # 3. Global Thermal Equilibrium Balance
        ln_t_pred = pred_clean[:, 0, :, :]
        ln_t_target = target_clean[:, 0, :, :]
        loss_thermal_balance = (torch.mean(ln_t_pred) - torch.mean(ln_t_target)) ** 2
        loss_thermal_balance = torch.nan_to_num(loss_thermal_balance, nan=0.0)
        metrics["loss_thermal_balance"] = loss_thermal_balance.item()
        total_loss += (self.lambda_thermal * loss_thermal_balance)

        # -----------------------------------------------------------------
        # 4. Surface Drag & Boundary Layer Friction Penalty (Levels 0..3)
        # Prevents U, V surface wind speed overestimation
        # -----------------------------------------------------------------
        if self.lambda_pbl_drag > 0.0 and static_topo is not None:
            u_sfc = pred_clean[:, 1, :4, :]
            v_sfc = pred_clean[:, 2, :4, :]
            lsm = static_topo[:, 1, :].unsqueeze(1)  # Land-sea mask [B, 1, Nodes]
            cd_drag = torch.where(lsm > 0.5, 0.005, 0.0015)
            loss_drag = torch.mean(cd_drag * (u_sfc**2 + v_sfc**2))
            loss_drag = torch.nan_to_num(loss_drag, nan=0.0)
            metrics["loss_pbl_drag"] = loss_drag.item()
            total_loss += (self.lambda_pbl_drag * loss_drag)

        # -----------------------------------------------------------------
        # 5. Planetary Boundary Layer Lapse Rate Constraint
        # Restricts dT/dz between Level 1 (2m) and Level 16 (2000m)
        # -----------------------------------------------------------------
        if self.lambda_lapse_rate > 0.0 and h_3d is not None:
            t_sfc = torch.exp(pred_clean[:, 0, 0, :] * self.std_ln_t + self.mu_ln_t)   # L01 (2m)
            t_pbl = torch.exp(pred_clean[:, 0, 15, :] * self.std_ln_t + self.mu_ln_t)  # L16 (2000m)

            h_sfc = h_3d_4d[:, 0, 0, :]
            h_pbl = h_3d_4d[:, 0, 15, :]
            dz_pbl = torch.clamp(h_pbl - h_sfc, min=100.0)

            dT_dz_pred = (t_pbl - t_sfc) / dz_pbl  # K/m
            loss_lapse = torch.mean(torch.relu(-dT_dz_pred - 0.012)**2) * 100.0
            loss_lapse = torch.nan_to_num(loss_lapse, nan=0.0)
            metrics["loss_lapse_rate"] = loss_lapse.item()
            total_loss += (self.lambda_lapse_rate * loss_lapse)

        # 6. Graph Laplacian Smoothness Penalty on Pressure
        p_pred = pred_clean[:, 6, :, :]
        src, dst = edge_index[0], edge_index[1]
        diff_p = p_pred[:, :, src] - p_pred[:, :, dst]
        loss_laplacian_p = torch.mean(diff_p ** 2)
        loss_laplacian_p = torch.nan_to_num(loss_laplacian_p, nan=0.0)
        metrics["loss_laplacian_p"] = loss_laplacian_p.item()
        total_loss += (self.lambda_laplacian_p * loss_laplacian_p)

        # 7. Geostrophic Dynamics Penalty
        if self.lambda_dyn > 0.0 and graph_mesh_ops is not None and hasattr(graph_mesh_ops, "Gx_sparse"):
            u_pred = pred_clean[:, 1, :, :]
            v_pred = pred_clean[:, 2, :, :]

            # Wrap in autocast(enabled=False) to ensure float32 autograd graph
            # with torch.cuda.amp.autocast(enabled=False):
            with torch.amp.autocast('cuda', enabled=False):
                p_pred_f32 = pred_clean[:, 6, :, :].float()
                dp_dx, dp_dy = graph_mesh_ops(p_pred_f32)

                f_coriolis = 2.0 * 7.2921e-5 * torch.sin(graph_mesh_ops.lat_deg * np.pi / 180.0).view(1, 1, -1).to(pred.device)
                f_coriolis = torch.where(torch.abs(f_coriolis) < 2e-5, torch.sign(f_coriolis) * 2e-5 + 2e-5, f_coriolis)

                ln_rho_unnorm = pred_clean[:, 5, :, :].float() * self.std_ln_rho + self.mu_ln_rho
                rho_pred = torch.clamp(torch.exp(torch.clamp(ln_rho_unnorm, min=-10.0, max=1.0)), min=1e-4, max=2.0)

                u_geo = torch.clamp(-1.0 / (rho_pred * f_coriolis) * dp_dy, min=-100.0, max=100.0)
                v_geo = torch.clamp(1.0 / (rho_pred * f_coriolis) * dp_dx, min=-100.0, max=100.0)

                loss_dyn = F.mse_loss(u_pred.float(), u_geo) + F.mse_loss(v_pred.float(), v_geo)

            loss_dyn = torch.nan_to_num(loss_dyn, nan=0.0)
            metrics["loss_dynamics_total"] = loss_dyn.item()
            total_loss += (self.lambda_dyn * loss_dyn)
        else:
            metrics["loss_dynamics_total"] = 0.0

        return total_loss, metrics
