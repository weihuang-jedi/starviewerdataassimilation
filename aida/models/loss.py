import torch
import torch.nn as nn
import torch.nn.functional as F


class AIDASurrogateLoss(nn.Module):
    """
    Physically-constrained regularized loss module for AIDA GNN Surrogate Model.
    Includes log-scale loss for specific humidity (q) and joint bias penalty.
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
        weight_q_log: float = 0.15,
        weight_joint_bias: float = 0.05,
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

        # 6. Specific Humidity (q) Log-Space Scale Penalty
        q_pred = pred[:, self.q_idx, :, :]
        q_target = target[:, self.q_idx, :, :]
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
