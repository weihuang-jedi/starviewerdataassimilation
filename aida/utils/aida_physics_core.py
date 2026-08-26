import torch
import torch.nn as nn
import numpy as np

class NonHydrostaticIcosahedralLoss(nn.Module):
    """
    Physically-constrained loss for z-coordinate non-hydrostatic atmospheric models.
    Operates on log-transformed density (ln_rho) and log-transformed temperature (ln_T).
    """
    def __init__(self, node_latitudes, R_gas=287.05, r_earth=6371000.0, omega=7.2921e-5):
        super(NonHydrostaticIcosahedralLoss, self).__init__()
        self.R = R_gas
        self.r_earth = r_earth
        self.omega = omega

        # Precompute Coriolis parameter (f) and Beta parameter (beta = df/dy)
        lats_rad = torch.deg2rad(node_latitudes)
        f_coriolis = 2.0 * omega * torch.sin(lats_rad)
        beta_coriolis = (2.0 * omega * torch.cos(lats_rad)) / r_earth

        self.register_buffer('f', f_coriolis.view(1, 1, -1))           # Shape: (1, 1, Nodes)
        self.register_buffer('beta', beta_coriolis.view(1, 1, -1))     # Shape: (1, 1, Nodes)
        self.register_buffer('is_equatorial', (torch.abs(node_latitudes) < 15.0).view(1, 1, -1))

    def forward(self, pred, target, mesh_level=0):
        """
        pred / target shape: (Batch, Vars, Levels, Nodes)
        Variables order: [ln_T, u, v, w, q, ln_rho]
        """
        loss_mse = torch.mean((pred - target) ** 2)

        ln_T   = pred[:, 0, :, :]
        u      = pred[:, 1, :, :]
        v      = pred[:, 2, :, :]
        ln_rho = pred[:, 5, :, :]

        T = torch.exp(ln_T)

        # ---------------------------------------------------------------
        # MESH 0: Global Mass Conservation & Global Mean Anchoring
        # ---------------------------------------------------------------
        # Mean global integrated log-density change must be near zero
        loss_mass = torch.abs(torch.mean(ln_rho) - torch.mean(target[:, 5, :, :]))

        if mesh_level == 0:
            return loss_mse + 0.1 * loss_mass

        # ---------------------------------------------------------------
        # MESH 1: Meridional Gradient & Latitude-Adaptive Geostrophic Loss
        # ---------------------------------------------------------------
        # Finite-difference approximation for spatial gradient along mesh nodes
        dT_dy    = T[:, :, 1:] - T[:, :, :-1]
        dlnrho_dy = ln_rho[:, :, 1:] - ln_rho[:, :, :-1]

        # Mid/High Latitudes Geostrophic Balance: f * u = -R*(dT/dy) - R*T*(dlnrho/dy)
        f_mid = self.f[:, :, 1:]
        u_mid = u[:, :, 1:]
        T_mid = T[:, :, 1:]
        rhs_mid = -self.R * dT_dy - self.R * T_mid * dlnrho_dy
        loss_geostrophic_mid = torch.mean(torch.abs(f_mid * u_mid - rhs_mid)[~self.is_equatorial[:, :, 1:]])

        # Equatorial Latitudes Beta-Plane Balance: beta * u = -R*(d²T/dy²) - R*T*(d²lnrho/dy²)
        if self.is_equatorial.any():
            d2T_dy2 = dT_dy[:, :, 1:] - dT_dy[:, :, :-1]
            d2lnrho_dy2 = dlnrho_dy[:, :, 1:] - dlnrho_dy[:, :, :-1]
            beta_eq = self.beta[:, :, 2:]
            u_eq = u[:, :, 2:]
            T_eq = T[:, :, 2:]
            rhs_eq = -self.R * d2T_dy2 - self.R * T_eq * d2lnrho_dy2
            loss_geostrophic_eq = torch.mean(torch.abs(beta_eq * u_eq - rhs_eq)[self.is_equatorial[:, :, 2:]])
        else:
            loss_geostrophic_eq = torch.tensor(0.0, device=pred.device)

        loss_mesh1 = loss_geostrophic_mid + loss_geostrophic_eq

        total_loss = loss_mse + 0.05 * loss_mass + 0.02 * loss_mesh1
        return total_loss
