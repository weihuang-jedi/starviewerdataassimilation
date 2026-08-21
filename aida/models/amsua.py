#!/usr/bin/env python3
"""
aida_combined_loss.py
---------------------
Combined PyTorch Loss Module for AIDA (Atmospheric Icosahedral Data Assimilation).

Integrates:
1. Differentiable AMSU-A Radiance Loss J_rad(x): Computes brightness temperature
   innovations y - H(x) in log-state space using channel-specific weighting functions.
2. Hybrid Dynamics Loss J_dyn(x): Enforces mid-latitude geostrophic balance and 
   tropical mass continuity across the M4 icosahedral mesh.
3. Log-State Constraint Loss J_thermo(x): Soft penalty verifying Ideal Gas Law 
   consistency (ln p = ln rho + ln R_d + ln T).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dry air gas constant
R_D = 287.058  # J / (kg * K)
LN_R_D = torch.log(torch.tensor(R_D, dtype=torch.float32))

# AMSU-A Channel Peak Pressures (hPa) and Scale Heights (hPa)
AMSUA_PEAK_P_HPA = torch.tensor([
    1000.0, 1000.0, 1000.0, 850.0, 700.0, 400.0, 250.0, 150.0,
    90.0, 50.0, 25.0, 10.0, 5.0, 2.0, 1000.0
], dtype=torch.float32)

AMSUA_SCALE_HPA = torch.tensor([
    300.0, 300.0, 250.0, 180.0, 140.0, 110.0, 80.0, 50.0,
    30.0, 18.0, 10.0, 4.0, 2.0, 1.0, 300.0
], dtype=torch.float32)

# Channel Observation Standard Errors (Kelvin)
AMSUA_OBS_ERR_K = torch.tensor([
    2.5, 2.2, 1.2, 0.6, 0.3, 0.25, 0.25, 0.25,
    0.25, 0.35, 0.55, 0.8, 1.2, 1.8, 3.5
], dtype=torch.float32)


class DifferentiableAMSUAOperator(nn.Module):
    """Differentiable Forward Radiative Transfer Operator H(x) for AMSU-A."""
    
    def __init__(self, peak_pressures: torch.Tensor = AMSUA_PEAK_P_HPA,
                       scale_heights: torch.Tensor = AMSUA_SCALE_HPA,
                       num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.register_buffer("peak_p", peak_pressures)
        self.register_buffer("scale_h", scale_heights)

    def forward(self, t_k: torch.Tensor, p_hpa: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            t_k:   Physical Temperature in Kelvin, shape [B, N_obs, N_levels]
            p_hpa: Physical Pressure in hPa, shape [B, N_obs, N_levels]
        Returns:
            tb_sim: Simulated Brightness Temp in Kelvin, shape [B, N_obs, 15]
        """
        # Expand dims for channel broadcasting: [B, N_obs, N_levels, 15]
        p_exp = p_hpa.unsqueeze(-1)
        log_p = torch.log(torch.clamp(p_exp, min=1e-3))
        log_p_peak = torch.log(self.peak_p)
        log_sigma = self.scale_h / self.peak_p

        # Gaussian weighting matrix in log-pressure space W_c(p)
        weights = torch.exp(-0.5 * ((log_p - log_p_peak) / log_sigma) ** 2)
        weights = weights / (torch.sum(weights, dim=-2, keepdim=True) + 1e-8)

        # Radiative Transfer Integration: \int T(p) * W_c(p) dp
        tb_sim = torch.sum(t_k.unsqueeze(-1) * weights, dim=-2)
        return tb_sim


class HybridDynamicsLoss(nn.Module):
    """
    Physical dynamics loss combining mid-latitude Geostrophic Balance 
    and Tropical Mass-Continuity (Low Divergence).
    """
    
    def __init__(self, grad_op: torch.Tensor, div_op: torch.Tensor, lats_deg: torch.Tensor):
        super().__init__()
        # Sparse or dense differential operators for M4 mesh
        self.register_buffer("grad_op", grad_op)
        self.register_buffer("div_op", div_op)
        
        # Latitude weighting vector (Cosine latitude blend between tropics and mid-latitudes)
        lats_rad = torch.radians(lats_deg)
        f_coriolis = 2.0 * 7.2921e-5 * torch.sin(lats_rad)
        
        # Blend factor: 0.0 at Equator (tropics -> divergence penalty), 1.0 at Mid-Lats (geostrophy)
        blend = torch.clamp(torch.sin(lats_rad) ** 2, min=0.0, max=1.0)
        
        self.register_buffer("f_coriolis", f_coriolis)
        self.register_buffer("blend", blend)

    def forward(self, u: torch.Tensor, v: torch.Tensor, ln_p: torch.Tensor, ln_rho: torch.Tensor) -> torch.Tensor:
        """
        Evaluates physical dynamic imbalances on the M4 mesh.
        u, v, ln_p, ln_rho shape: [B, N_nodes, N_levels]
        """
        # 1. Mass Continuity / Divergence Penalty (dominant in tropics)
        # Div(V) = dU/dx + dV/dy
        div_u = torch.matmul(self.div_op, u)
        div_v = torch.matmul(self.div_op, v)
        divergence = div_u + div_v
        tropical_div_loss = torch.mean((1.0 - self.blend.unsqueeze(-1)) * (divergence ** 2))

        # 2. Geostrophic Balance Penalty (dominant in mid-latitudes)
        # f * v = (1 / rho) * dp/dx  =>  f * v = R_d * T * d(ln_p)/dx
        dp_dx = torch.matmul(self.grad_op, ln_p)
        dp_dy = torch.matmul(self.grad_op, ln_p)
        
        # Acceleration residuals
        f_exp = self.f_coriolis.unsqueeze(-1)
        geostrophic_u_res = f_exp * v + dp_dx
        geostrophic_v_res = f_exp * u - dp_dy
        
        midlat_geo_loss = torch.mean(
            self.blend.unsqueeze(-1) * (geostrophic_u_res ** 2 + geostrophic_v_res ** 2)
        )

        return tropical_div_loss + midlat_geo_loss


class AIDALossEngine(nn.Module):
    """
    Master AIDA Loss Function incorporating Radiance, Dynamics, 
    Thermodynamic Log-State Consistency, and Pattern Skill Losses.
    """

    def __init__(
        self,
        grad_op: torch.Tensor,
        div_op: torch.Tensor,
        lats_deg: torch.Tensor,
        w_rad_amsua: float = 1.0,
        lambda_dyn: float = 0.1,
        w_thermo: float = 0.05,
        w_acc: float = 0.2
    ):
        super().__init__()
        self.h_amsua = DifferentiableAMSUAOperator()
        self.dynamics_loss = HybridDynamicsLoss(grad_op, div_op, lats_deg)
        
        self.w_rad_amsua = w_rad_amsua
        self.lambda_dyn = lambda_dyn
        self.w_thermo = w_thermo
        self.w_acc = w_acc
        
        self.register_buffer("obs_err_k", AMSUA_OBS_ERR_K)

    def forward(
        self,
        state_pred: dict,
        y_amsua: torch.Tensor,
        amsua_mask: torch.Tensor,
        state_target: dict = None
    ) -> dict:
        """
        Parameters:
            state_pred: Dict containing M4 log-state fields:
                        ['ln_T', 'ln_p', 'ln_rho', 'u', 'v'] -> [B, N_nodes, N_levels]
            y_amsua:    Observed AMSU-A Brightness Temperatures -> [B, N_nodes, 15]
            amsua_mask: Boolean mask for valid observation points -> [B, N_nodes]
            state_target: (Optional) Ground truth target for ACC pattern correlation.
        """
        ln_t = state_pred["ln_T"]
        ln_p = state_pred["ln_p"]
        ln_rho = state_pred["ln_rho"]
        u = state_pred["u"]
        v = state_pred["v"]

        # ----------------------------------------------------------------------
        # 1. AMSU-A Radiance Observation Loss J_rad(x)
        # ----------------------------------------------------------------------
        t_k = torch.exp(ln_t)                  # K
        p_hpa = torch.exp(ln_p) / 100.0        # Pa -> hPa
        
        # Evaluate H(x) forward operator
        tb_sim = self.h_amsua(t_k, p_hpa)      # [B, N_nodes, 15]

        # Normalized Innovation Square: ((y - H(x)) / sigma)^2
        innovation = (y_amsua - tb_sim) / self.obs_err_k
        squared_innov = innovation ** 2

        # Apply spatial observation mask
        if amsua_mask is not None:
            mask_exp = amsua_mask.unsqueeze(-1).expand_as(squared_innov)
            j_rad = torch.sum(squared_innov * mask_exp) / (torch.sum(mask_exp) + 1e-8)
        else:
            j_rad = torch.mean(squared_innov)

        # ----------------------------------------------------------------------
        # 2. Hybrid Physical Dynamics Loss J_dyn(x)
        # ----------------------------------------------------------------------
        j_dyn = self.dynamics_loss(u, v, ln_p, ln_rho)

        # ----------------------------------------------------------------------
        # 3. Log-State Ideal Gas Law Constraint J_thermo(x)
        # Enforces exact relation: ln p - (ln rho + ln R_d + ln T) = 0
        # ----------------------------------------------------------------------
        thermo_residual = ln_p - (ln_rho + LN_R_D + ln_t)
        j_thermo = torch.mean(thermo_residual ** 2)

        # ----------------------------------------------------------------------
        # 4. Optional Spatial Pattern Skill Loss (ACC Anomaly Loss)
        # ----------------------------------------------------------------------
        j_acc = torch.tensor(0.0, device=ln_t.device)
        if state_target is not None:
            # Pattern correlation loss across pressure field
            p_pred_mean = ln_p - torch.mean(ln_p, dim=1, keepdim=True)
            p_targ_mean = state_target["ln_p"] - torch.mean(state_target["ln_p"], dim=1, keepdim=True)
            
            num = torch.sum(p_pred_mean * p_targ_mean, dim=1)
            den = torch.sqrt(torch.sum(p_pred_mean ** 2, dim=1) * torch.sum(p_targ_mean ** 2, dim=1) + 1e-8)
            acc = num / den
            j_acc = torch.mean(1.0 - acc)  # Minimize 1 - ACC

        # Total Composite Objective Function
        total_loss = (
            self.w_rad_amsua * j_rad +
            self.lambda_dyn * j_dyn +
            self.w_thermo * j_thermo +
            self.w_acc * j_acc
        )

        return {
            "loss": total_loss,
            "j_rad": j_rad.detach(),
            "j_dyn": j_dyn.detach(),
            "j_thermo": j_thermo.detach(),
            "j_acc": j_acc.detach() if isinstance(j_acc, torch.Tensor) else j_acc
        }
