#!/usr/bin/env python3
"""
models/loss.py
--------------
Multi-Component Physical & Satellite Radiance Loss Engine for AIDA GNN Surrogate Model.
Includes Geostrophic/Tropical Dynamics, Pressure Laplacian, Asymmetric Barriers,
Thermodynamic State Equation, and Differentiable Radiance Innovation Losses (AMSU-A + IASI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.amsua import DifferentiableAMSUAOperator
from models.iasi import DifferentiableIASIOperator
from models.hms import DifferentiableHMSOperator
from models.atms import DifferentiableATMSOperator
from models.cris import DifferentiableCrISOperator
from models.seviri import DifferentiableSEVIRIOperator
from models.gsrasr import DifferentiableGSRASROperator
from models.gsrcsr import DifferentiableGSRCSR0perator


def build_icosahedral_differential_operators(
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    edge_index: torch.Tensor,
    R: float = 6.371e6
):
    """
    Pre-computes sparse gradient operator matrices (Gx, Gy) for an icosahedral mesh.
    """
    N = lat_deg.shape[0]
    lat_rad = torch.deg2rad(lat_deg)
    lon_rad = torch.deg2rad(lon_deg)

    X = R * torch.cos(lat_rad) * torch.cos(lon_rad)
    Y = R * torch.cos(lat_rad) * torch.sin(lon_rad)
    Z = R * torch.sin(lat_rad)

    ex = torch.stack([-torch.sin(lon_rad), torch.cos(lon_rad), torch.zeros_like(lon_rad)], dim=1)
    ey = torch.stack([
        -torch.sin(lat_rad) * torch.cos(lon_rad),
        -torch.sin(lat_rad) * torch.sin(lon_rad),
        torch.cos(lat_rad)
    ], dim=1)

    src, dst = edge_index[0], edge_index[1]

    row_indices = []
    col_indices = []
    values_x = []
    values_y = []

    for i in range(N):
        neighbors = dst[src == i]
        if len(neighbors) == 0:
            continue

        P_i = torch.tensor([X[i], Y[i], Z[i]])
        ex_i, ey_i = ex[i], ey[i]

        dx_local = []
        dy_local = []
        valid_nbrs = []

        for j in neighbors:
            P_j = torch.tensor([X[j], Y[j], Z[j]])
            delta_P = P_j - P_i

            dx = torch.dot(delta_P, ex_i).item()
            dy = torch.dot(delta_P, ey_i).item()

            dx_local.append(dx)
            dy_local.append(dy)
            valid_nbrs.append(j.item())

        A = np.column_stack([dx_local, dy_local])
        dist_sq = np.array(dx_local)**2 + np.array(dy_local)**2
        W = np.diag(1.0 / np.maximum(dist_sq, 1e-6))

        try:
            AtWA = A.T @ W @ A
            pinv = np.linalg.inv(AtWA) @ A.T @ W

            weights_x = pinv[0, :]
            weights_y = pinv[1, :]

            sum_wx, sum_wy = 0.0, 0.0
            for k, nbr in enumerate(valid_nbrs):
                row_indices.append(i)
                col_indices.append(nbr)
                values_x.append(weights_x[k])
                values_y.append(weights_y[k])
                sum_wx += weights_x[k]
                sum_wy += weights_y[k]

            row_indices.append(i)
            col_indices.append(i)
            values_x.append(-sum_wx)
            values_y.append(-sum_wy)

        except np.linalg.LinAlgError:
            continue

    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values_x = torch.tensor(values_x, dtype=torch.float32)
    values_y = torch.tensor(values_y, dtype=torch.float32)

    Gx_sparse = torch.sparse_coo_tensor(indices, values_x, size=(N, N)).to_sparse_csr()
    Gy_sparse = torch.sparse_coo_tensor(indices, values_y, size=(N, N)).to_sparse_csr()

    return Gx_sparse, Gy_sparse


class M4MeshOperators(nn.Module):
    """
    Executes sparse differentiable spatial operators (Grad, Div) on M4 Mesh tensor batches.
    """
    def __init__(self, Gx_sparse: torch.Tensor, Gy_sparse: torch.Tensor, lat_deg: torch.Tensor):
        super().__init__()
        self.register_buffer("Gx", Gx_sparse.to(torch.float32))
        self.register_buffer("Gy", Gy_sparse.to(torch.float32))

        lat_rad = torch.deg2rad(lat_deg.to(torch.float32))
        self.register_buffer("latitudes_rad", lat_rad)

    def compute_gradient(self, p: torch.Tensor):
        p_dtype = self.Gx.dtype
        if p.dim() == 1:
            p_in = p.unsqueeze(1).to(p_dtype)
            dp_dx = torch.sparse.mm(self.Gx, p_in).squeeze(1)
            dp_dy = torch.sparse.mm(self.Gy, p_in).squeeze(1)
        else:
            p_in = p.T.to(p_dtype)
            dp_dx = torch.sparse.mm(self.Gx, p_in).T
            dp_dy = torch.sparse.mm(self.Gy, p_in).T

        return dp_dx.to(p.dtype), dp_dy.to(p.dtype)

    def compute_divergence(self, u: torch.Tensor, v: torch.Tensor):
        du_dx, _ = self.compute_gradient(u)
        _, dv_dy = self.compute_gradient(v)
        return du_dx + dv_dy


class HybridDynamicsLoss(nn.Module):
    """
    Hybrid Atmospheric Dynamics Loss for Global Spherical Graph Neural Networks (AIDA).
    """
    def __init__(
        self,
        omega: float = 7.292115e-5,
        rho_ref: float = 1.225,
        lat_trans_center_deg: float = 15.0,
        lat_trans_width_deg: float = 10.0,
        eps_f: float = 1e-5
    ):
        super().__init__()
        self.omega = omega
        self.rho_ref = rho_ref
        self.lat_trans_center = torch.tensor(lat_trans_center_deg * torch.pi / 180.0)
        self.lat_trans_width = torch.tensor(lat_trans_width_deg * torch.pi / 180.0)
        self.eps_f = eps_f

    def _compute_latitude_weights(self, lat_rad: torch.Tensor):
        abs_lat = torch.abs(lat_rad)
        lat_min = self.lat_trans_center - (self.lat_trans_width / 2.0)
        s = torch.clamp((abs_lat - lat_min) / self.lat_trans_width, 0.0, 1.0)
        w_geo = 0.5 * (1.0 - torch.cos(torch.pi * s))
        w_trop = 1.0 - w_geo
        return w_geo, w_trop

    def forward(
        self,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
        p_pred: torch.Tensor,
        lat_rad: torch.Tensor,
        grad_p_x: torch.Tensor,
        grad_p_y: torch.Tensor,
        div_v: torch.Tensor
    ) -> dict:
        w_geo, w_trop = self._compute_latitude_weights(lat_rad)
        f = 2.0 * self.omega * torch.sin(lat_rad)

        f_sign = torch.sign(f)
        f_sign = torch.where(f_sign == 0, torch.ones_like(f_sign), f_sign)
        f_safe = torch.where(torch.abs(f) < self.eps_f, self.eps_f * f_sign, f)

        u_geo_target = -1.0 / (self.rho_ref * f_safe) * grad_p_y
        v_geo_target =  1.0 / (self.rho_ref * f_safe) * grad_p_x

        loss_u_geo = torch.square(u_pred - u_geo_target)
        loss_v_geo = torch.square(v_pred - v_geo_target)
        loss_geo_per_node = loss_u_geo + loss_v_geo

        loss_geostrophic = torch.sum(w_geo * loss_geo_per_node) / (torch.sum(w_geo) + 1e-8)

        loss_div_per_node = torch.square(div_v)
        loss_tropical = torch.sum(w_trop * loss_div_per_node) / (torch.sum(w_trop) + 1e-8)

        loss_total = loss_geostrophic + loss_tropical

        return {
            "loss_dynamics_total": loss_total,
            "loss_geostrophic": loss_geostrophic,
            "loss_tropical_div": loss_tropical
        }


class AIDASurrogateLoss(nn.Module):
    """
    Unified AIDASurrogateLoss module targeting physical constraints,
    spatial pattern correlations, and satellite radiance innovations.
    """
    def __init__(
        self,
        t_idx: int = 0,
        u_idx: int = 1,
        v_idx: int = 2,
        q_idx: int = 4,
        p_idx: int = 6,
        sharp_var_indices: list[int] = [0, 1, 2, 4, 6],
        weight_mse: float = 1.0,
        w_mse: float = 1.0,
        lambda_laplacian_p: float = 0.18,
        weight_grad_state: float = 0.25,
        lambda_p_acc: float = 0.15,
        lambda_asym_p: float = 0.35,
        weight_state_eq: float = 0.12,
        weight_q_log: float = 0.25,
        w_rad_amsua: float = 0.01,
        w_rad_iasi: float = 0.01,
        w_rad_hms: float = 0.01,
        w_rad_atms: float = 0.01,  # Added ATMS loss weight
        w_rad_cris: float = 0.01,  # Added CrIS loss weight
        w_rad_seviri: float = 0.01,  # Added SEVIRI loss weight
        w_rad_gsrasr: float = 0.01,  # Added GSRASR loss weight
        w_rad_gsrcsr: float = 0.01,  # Added GSRCSR loss weight
        lambda_asym_q: float = 0.50,
        lambda_dyn: float = 0.01,
        w_dyn: float = 0.01,
        q_floor: float = 1e-7,
        weight_joint_bias: float = 0.10,
        tau_min_p: float = 0.08,
        mu_ln_t: float = 5.5, std_ln_t: float = 0.2,
        mu_ln_rho: float = -0.5, std_ln_rho: float = 0.5,
        mu_ln_p: float = 11.5, std_ln_p: float = 0.3,
        R_d: float = 287.058,
        num_levels: int = 32,
        **kwargs
    ):
        super().__init__()
        self.t_idx = t_idx
        self.u_idx = u_idx
        self.v_idx = v_idx
        self.q_idx = q_idx
        self.p_idx = p_idx
        self.sharp_var_indices = sharp_var_indices

        self.weight_mse = w_mse if w_mse != 1.0 else weight_mse
        self.lambda_laplacian_p = lambda_laplacian_p
        self.weight_grad_state = weight_grad_state
        self.lambda_p_acc = lambda_p_acc
        self.lambda_asym_p = lambda_asym_p
        self.weight_state_eq = weight_state_eq
        self.weight_q_log = weight_q_log
        self.lambda_asym_q = lambda_asym_q
        self.lambda_dyn = lambda_dyn if lambda_dyn != 0.01 else w_dyn
        self.q_floor = q_floor
        self.weight_joint_bias = weight_joint_bias
        self.tau_min_p = tau_min_p

        self.w_rad_amsua = w_rad_amsua
        self.w_rad_iasi = w_rad_iasi
        self.w_rad_hms = w_rad_hms
        self.w_rad_atms = w_rad_atms
        self.w_rad_cris = w_rad_cris
        self.w_rad_seviri = w_rad_seviri
        self.w_rad_gsrasr = w_rad_gsrasr
        self.w_rad_gsrcsr = w_rad_gsrcsr

        self.hybrid_dyn = HybridDynamicsLoss(lat_trans_center_deg=15.0)

        self.mu_ln_t, self.std_ln_t = mu_ln_t, std_ln_t
        self.mu_ln_rho, self.std_ln_rho = mu_ln_rho, std_ln_rho
        self.mu_ln_p, self.std_ln_p = mu_ln_p, std_ln_p
        self.register_buffer("ln_R_d", torch.log(torch.tensor(R_d)))

        self.mse_fn = nn.MSELoss()

        self.amsua_loss = DifferentiableAMSUAOperator(num_levels=num_levels)
        self.iasi_loss = DifferentiableIASIOperator(num_levels=num_levels)
        self.hms_loss = DifferentiableHMSOperator(num_levels=num_levels)
        self.atms_loss = DifferentiableATMSOperator(num_levels=num_levels)
        self.cris_loss = DifferentiableCrISOperator(num_levels=num_levels)
        self.seviri_loss = DifferentiableSEVIRIOperator(num_levels=num_levels)
        self.gsrasr_loss = DifferentiableGSRASROperator(num_levels=num_levels)
        self.gsrcsr_loss = DifferentiableGSRCSROperator(num_levels=num_levels)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor,
        graph_mesh_ops: M4MeshOperators = None
    ) -> tuple[torch.Tensor, dict[str, float]]:

        B, V, L, N = pred.shape
        src, dst = edge_index[0], edge_index[1]

        # 1. Base Data Fidelity Loss
        loss_mse = self.mse_fn(pred, target)

        # 2. 2nd-Order Graph Laplacian Pressure Penalty
        p_pred = pred[:, self.p_idx, :, :]  # [B, L, N]

        p_neighbor_sum = torch.zeros_like(p_pred)
        dst_expanded = dst.view(1, 1, -1).expand(B, L, -1)
        p_neighbor_sum.scatter_add_(2, dst_expanded, p_pred.index_select(2, src))

        deg = torch.zeros(N, device=p_pred.device, dtype=p_pred.dtype)
        deg.index_add_(0, dst, torch.ones_like(src, dtype=p_pred.dtype))
        deg = deg.view(1, 1, N)
        deg_clamped = torch.clamp(deg, min=1.0)

        p_neighbor_avg = p_neighbor_sum / deg_clamped
        loss_laplacian_p = torch.mean(torch.square(p_pred - p_neighbor_avg))

        # 3. Pressure Spatial Pattern Correlation Loss (Direct ACC optimization)
        p_target = target[:, self.p_idx, :, :]
        p_pred_anon = p_pred - torch.mean(p_pred, dim=-1, keepdim=True)
        p_true_anon = p_target - torch.mean(p_target, dim=-1, keepdim=True)

        cos_sim_p = F.cosine_similarity(p_pred_anon.flatten(1), p_true_anon.flatten(1), dim=-1)
        loss_p_acc = torch.mean(1.0 - cos_sim_p)

        # 4. State Gradient Matching Loss (Excludes w)
        sharp_pred = pred[:, self.sharp_var_indices, :, :]
        sharp_target = target[:, self.sharp_var_indices, :, :]

        diff_pred = sharp_pred.index_select(3, src) - sharp_pred.index_select(3, dst)
        diff_target = sharp_target.index_select(3, src) - sharp_target.index_select(3, dst)
        loss_grad_state = torch.mean(torch.abs(diff_pred - diff_target))

        # 5. Asymmetric Barrier Penalty for Low Pressure Spikes
        p_violation = F.relu(self.tau_min_p - p_pred)
        loss_asym_p = torch.mean(torch.square(p_violation))

        # 6. Ideal Gas Thermodynamic Coupling Residual
        ln_T_phys = torch.clamp(pred[:, self.t_idx, :, :] * self.std_ln_t + self.mu_ln_t, min=4.95, max=6.0)
        ln_rho_phys = torch.clamp(pred[:, 5, :, :] * self.std_ln_rho + self.mu_ln_rho, min=-15.0, max=2.0)
        ln_p_phys = torch.clamp(pred[:, self.p_idx, :, :] * self.std_ln_p + self.mu_ln_p, min=-5.0, max=13.0)

        state_eq_residual = ln_p_phys - (ln_rho_phys + self.ln_R_d + ln_T_phys)
        loss_state_eq = torch.mean(torch.abs(state_eq_residual))

        # 7. Specific Humidity (q) Loss
        q_raw_pred = pred[:, self.q_idx, :, :]
        q_raw_target = target[:, self.q_idx, :, :]

        q_pred_clamped = torch.clamp(q_raw_pred, min=self.q_floor)
        q_true_clamped = torch.clamp(q_raw_target, min=self.q_floor)
        loss_q_log = torch.mean(torch.abs(torch.log(q_pred_clamped) - torch.log(q_true_clamped)))

        q_violation = F.relu(self.q_floor - q_raw_pred)
        loss_asym_q = torch.mean(torch.square(q_violation))

        # 8. Enhanced Joint Mean Bias Penalty
        bias_p = torch.abs(torch.mean(p_pred) - torch.mean(target[:, self.p_idx, :, :]))
        bias_t = torch.abs(torch.mean(pred[:, self.t_idx, :, :]) - torch.mean(target[:, self.t_idx, :, :]))
        bias_q = torch.abs(torch.mean(pred[:, self.q_idx, :, :]) - torch.mean(target[:, self.q_idx, :, :]))

        loss_joint_bias = bias_p + 5.0 * bias_t + 50.0 * bias_q

        # 9. Hybrid Dynamics Loss Integration
        if graph_mesh_ops is not None:
            u_2d = pred[:, self.u_idx, :, :].mean(dim=1)
            v_2d = pred[:, self.v_idx, :, :].mean(dim=1)
            p_2d = pred[:, self.p_idx, :, :].mean(dim=1)

            grad_px, grad_py = graph_mesh_ops.compute_gradient(p_2d)
            div_v = graph_mesh_ops.compute_divergence(u_2d, v_2d)

            dyn_losses = self.hybrid_dyn(
                u_pred=u_2d,
                v_pred=v_2d,
                p_pred=p_2d,
                lat_rad=graph_mesh_ops.latitudes_rad,
                grad_p_x=grad_px,
                grad_p_y=grad_py,
                div_v=div_v
            )
            loss_dyn = dyn_losses['loss_dynamics_total']
            loss_geo = dyn_losses['loss_geostrophic'].item()
            loss_trop = dyn_losses['loss_tropical_div'].item()
        else:
            loss_dyn = torch.tensor(0.0, device=pred.device)
            loss_geo = 0.0
            loss_trop = 0.0

        # Total Weighted Combination
        total_loss = (
            self.weight_mse * loss_mse
            + self.lambda_laplacian_p * loss_laplacian_p
            + self.lambda_p_acc * loss_p_acc
            + self.weight_grad_state * loss_grad_state
            + self.lambda_asym_p * loss_asym_p
            + self.weight_state_eq * loss_state_eq
            + self.weight_q_log * loss_q_log
            + self.lambda_asym_q * loss_asym_q
            + self.lambda_dyn * loss_dyn
            + self.weight_joint_bias * loss_joint_bias
        )

        loss_metrics = {
            "loss_total": total_loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_laplacian_p": loss_laplacian_p.item(),
            "loss_p_acc": loss_p_acc.item(),
            "loss_grad_state": loss_grad_state.item(),
            "loss_asym_p": loss_asym_p.item(),
            "loss_state_eq": loss_state_eq.item(),
            "loss_q_log": loss_q_log.item(),
            "loss_asym_q": loss_asym_q.item(),
            "loss_joint_bias": loss_joint_bias.item(),
            "loss_dynamics_total": loss_dyn.item() if isinstance(loss_dyn, torch.Tensor) else loss_dyn,
            "loss_geostrophic": loss_geo,
            "loss_tropical_div": loss_trop,
        }

        return total_loss, loss_metrics
