#!/usr/bin/env python3
"""
Multivariate 3D-Var Pipeline with Weak Physical Constraints
-----------------------------------------------------------
Assimilates 5 Variables: [p, t, u, v, q]

Features:
  1. Geostrophic Balance Constraint (couples [p] to [u, v] via pressure gradients).
  2. Hydrostatic Balance Constraint (couples vertical [p] gradients to [t]).
  3. Physical Moisture Bounding (prevents negative specific humidity q).
  4. Spatial Correlation & 3.5-sigma Quality Control.
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cholesky
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

try:
    from nnja_ai import DataCatalog
    HAS_NNJA = True
except ImportError:
    HAS_NNJA = False

warnings.filterwarnings("ignore", category=UserWarning, module="nnja_ai")

VAR_LIST = ["p", "t", "u", "v", "q"]
NNJA_VAR_MAP = {
    "p": ["PRLC", "PRES", "PRSE"],
    "t": ["TMDB", "TMP"],
    "u": ["UGRD", "UWND", "U_WIND"],
    "v": ["VGRD", "VWND", "V_WIND"],
    "q": ["SPFH", "Q", "HUMI"],
}

DEFAULT_OBS_ERRORS = {
    "p": 2.0,     # hPa
    "t": 1.5,     # K
    "u": 2.0,     # m/s
    "v": 2.0,     # m/s
    "q": 0.002,   # kg/kg
}

SIGMA_HORIZONTAL = {
    "p": 3.0,
    "t": 2.5,
    "u": 3.5,
    "v": 3.5,
    "q": 2.0,
}

# Physical Constants for Weak Constraints
G_ACCEL = 9.80665       # m/s^2
R_DRY = 287.058         # J/(kg K)
OMEGA_EARTH = 7.2921e-5 # rad/s
EARTH_RADIUS = 6.371e6  # meters

# Weights for weak physical constraint penalties (J_c)
GAMMA_GEO = 0.15   # Geostrophic coupling weight
GAMMA_HYDRO = 0.10 # Hydrostatic coupling weight


# ==============================================================================
# 1. ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 5-Var 3D-Var with Weak Geostrophic & Hydrostatic Constraints."
    )
    parser.add_argument("--gfs_file", type=str, required=True, help="Path to GFS background NetCDF.")
    parser.add_argument("--bmatrix_file", type=str, required=True, help="Path to B-Matrix NetCDF.")
    parser.add_argument("--time", type=str, default="2023-07-01", help="Assimilation date YYYY-MM-DD.")
    parser.add_argument("--output", type=str, default="gfs_5var_3dvar_constrained.nc", help="Output NetCDF.")
    parser.add_argument("--max_iter", type=int, default=40, help="Max optimization iterations.")
    return parser.parse_args()


# ==============================================================================
# 2. DATA LOADING & COVARIANCE SETUP
# ==============================================================================
def load_gfs_and_bmatrix(gfs_path: str, bmatrix_path: str):
    print(f"[1/5] Loading Background State from: {gfs_path}")
    x_b_dict = {}
    sigma_b_dict = {}
    L_v_dict = {}

    with xr.open_dataset(gfs_path) as ds_gfs:
        heights = ds_gfs["height"].values.astype(np.float64)
        lats = ds_gfs["latitude"].values.astype(np.float64)
        lons = ds_gfs["longitude"].values.astype(np.float64)

        lat_asc = np.all(np.diff(lats) > 0)
        if not lat_asc:
            lats = lats[::-1]

        for var in VAR_LIST:
            val = ds_gfs[var].values.astype(np.float64)
            if not lat_asc:
                val = np.flip(val, axis=1)
            x_b_dict[var] = val

    print(f"[2/5] Loading B-Matrix Error Statistics from: {bmatrix_path}")
    with xr.open_dataset(bmatrix_path) as ds_b:
        for var in VAR_LIST:
            var_3d = ds_b[f"{var}_var"].values.astype(np.float64)
            if not lat_asc:
                var_3d = np.flip(var_3d, axis=1)

            sigma_b_dict[var] = np.sqrt(np.maximum(var_3d, 1e-8))

            vert_corr = ds_b[f"{var}_vert_corr"].values.astype(np.float64)
            vert_corr += np.eye(vert_corr.shape[0]) * 1e-6
            L_v_dict[var] = cholesky(vert_corr, lower=True)

    return x_b_dict, sigma_b_dict, L_v_dict, heights, lats, lons


# ==============================================================================
# 3. NNJA-AI CATALOG INGESTION
# ==============================================================================
def fetch_nnja_observations(time_str: str, heights, lats, lons, x_b_dict):
    obs_dict = {}

    if not HAS_NNJA:
        print("Warning: 'nnja_ai' not installed. Generating domain fallback obs.")
        return generate_synthetic_obs_all(heights, lats, lons, x_b_dict)

    date_part = time_str.split("T")[0]
    print(f"\n[3/5] Querying NNJA-AI Catalog ('conv-adpupa-NC002001') for: {date_part}...")

    try:
        catalog = DataCatalog(mirror="gcp_brightband")
        ds_upr = catalog["conv-adpupa-NC002001"]
        subset = ds_upr.sel(time=date_part)
        df_upr = subset.load_dataset(backend="pandas")
    except Exception as e:
        print(f"NNJA access notice: {e}. Falling back to domain synthetic obs.")
        return generate_synthetic_obs_all(heights, lats, lons, x_b_dict)

    if df_upr.empty:
        return generate_synthetic_obs_all(heights, lats, lons, x_b_dict)

    grid_lon_min, grid_lon_max = lons.min(), lons.max()
    if grid_lon_min < 0:
        df_upr["LON"] = np.where(df_upr["LON"] > 180, df_upr["LON"] - 360, df_upr["LON"])
    else:
        df_upr["LON"] = np.where(df_upr["LON"] < 0, df_upr["LON"] + 360, df_upr["LON"])

    lat_min, lat_max = lats.min(), lats.max()
    h_min, h_max = heights.min(), heights.max()

    for var in VAR_LIST:
        keywords = NNJA_VAR_MAP[var]
        match_cols = [c for c in df_upr.columns if any(k in c for k in keywords)]

        records = []
        for _, row in df_upr.iterrows():
            lat, lon = row["LAT"], row["LON"]
            if not (lat_min <= lat <= lat_max and grid_lon_min <= lon <= grid_lon_max):
                continue

            for col in match_cols:
                val = row[col]
                if pd.notna(val) and not np.isnan(val):
                    p_level = 850.0
                    for p in [1000, 925, 850, 700, 500, 300, 250, 200, 100]:
                        if str(p) in col:
                            p_level = float(p)
                            break

                    h_approx = 44330.0 * (1.0 - (p_level / 1013.25)**0.1903)
                    h_clamped = np.clip(h_approx, h_min, h_max)

                    if var == "t" and val < 150.0:
                        val += 273.15

                    records.append({
                        "height": h_clamped,
                        "latitude": lat,
                        "longitude": lon,
                        "observation_value": float(val),
                        "observation_error": DEFAULT_OBS_ERRORS[var],
                    })

        df_var = pd.DataFrame(records)

        if df_var.empty:
            obs_dict[var] = generate_single_synth_obs(heights, lats, lons, x_b_dict[var], var)
        else:
            obs_h = df_var["height"].values
            obs_lat = df_var["latitude"].values
            obs_lon = df_var["longitude"].values
            obs_y = df_var["observation_value"].values
            obs_err = df_var["observation_error"].values

            interp = RegularGridInterpolator(
                (heights, lats, lons), x_b_dict[var], bounds_error=False, fill_value=None
            )
            x_b_at_obs = interp(np.column_stack((obs_h, obs_lat, obs_lon)))

            innovations = np.abs(obs_y - x_b_at_obs)
            qc_mask = innovations <= (3.5 * obs_err)

            obs_dict[var] = (
                obs_h[qc_mask],
                obs_lat[qc_mask],
                obs_lon[qc_mask],
                obs_y[qc_mask],
                obs_err[qc_mask],
            )
            print(f"Ingested {np.sum(qc_mask)} valid observations for '{var}'.")

    return obs_dict


def generate_single_synth_obs(heights, lats, lons, x_b_var, var_name, n_obs=150):
    np.random.seed(42)
    obs_h = np.random.uniform(heights.min(), heights.max(), n_obs)
    obs_lat = np.random.uniform(lats.min(), lats.max(), n_obs)
    obs_lon = np.random.uniform(lons.min(), lons.max(), n_obs)

    interp = RegularGridInterpolator((heights, lats, lons), x_b_var, bounds_error=False, fill_value=None)
    obs_pts = np.column_stack((obs_h, obs_lat, obs_lon))
    err_std = DEFAULT_OBS_ERRORS[var_name]
    obs_y = interp(obs_pts) + np.random.normal(0.0, err_std, n_obs)
    errs = np.full(n_obs, err_std, dtype=np.float64)
    return obs_h, obs_lat, obs_lon, obs_y, errs


def generate_synthetic_obs_all(heights, lats, lons, x_b_dict):
    obs_dict = {}
    for var in VAR_LIST:
        obs_dict[var] = generate_single_synth_obs(heights, lats, lons, x_b_dict[var], var)
    return obs_dict


# ==============================================================================
# 4. OBSERVATION OPERATOR
# ==============================================================================
class ObservationOperator3D:
    def __init__(self, heights, lats, lons, obs_heights, obs_lats, obs_lons):
        self.heights = heights
        self.lats = lats
        self.lons = lons
        self.obs_pts = np.column_stack((obs_heights, obs_lats, obs_lons))
        self.grid_shape = (len(heights), len(lats), len(lons))

    def H(self, grid_state):
        interp = RegularGridInterpolator(
            (self.heights, self.lats, self.lons),
            grid_state,
            bounds_error=False,
            fill_value=None,
        )
        return interp(self.obs_pts)

    def H_adjoint(self, obs_residuals):
        adj_grid = np.zeros(self.grid_shape, dtype=np.float64)
        for k_obs, (h, lat, lon) in enumerate(self.obs_pts):
            i_h = np.clip(np.searchsorted(self.heights, h) - 1, 0, len(self.heights) - 2)
            i_lat = np.clip(np.searchsorted(self.lats, lat) - 1, 0, len(self.lats) - 2)
            i_lon = np.clip(np.searchsorted(self.lons, lon) - 1, 0, len(self.lons) - 2)

            wh = (h - self.heights[i_h]) / (self.heights[i_h + 1] - self.heights[i_h])
            wlat = (lat - self.lats[i_lat]) / (self.lats[i_lat + 1] - self.lats[i_lat])
            wlon = (lon - self.lons[i_lon]) / (self.lons[i_lon + 1] - self.lons[i_lon])

            res = obs_residuals[k_obs]

            adj_grid[i_h, i_lat, i_lon] += (1 - wh) * (1 - wlat) * (1 - wlon) * res
            adj_grid[i_h + 1, i_lat, i_lon] += wh * (1 - wlat) * (1 - wlon) * res
            adj_grid[i_h, i_lat + 1, i_lon] += (1 - wh) * wlat * (1 - wlon) * res
            adj_grid[i_h, i_lat, i_lon + 1] += (1 - wh) * (1 - wlat) * wlon * res
            adj_grid[i_h + 1, i_lat + 1, i_lon] += wh * wlat * (1 - wlon) * res
            adj_grid[i_h + 1, i_lat, i_lon + 1] += wh * (1 - wlat) * wlon * res
            adj_grid[i_h, i_lat + 1, i_lon + 1] += (1 - wh) * wlat * wlon * res
            adj_grid[i_h + 1, i_lat + 1, i_lon + 1] += wh * wlat * wlon * res

        return adj_grid


# ==============================================================================
# 5. B-MATRIX OPERATORS & WEAK CONSTRAINT CALCULATOR
# ==============================================================================
def apply_B_half_var(v_single, L_v, sigma_b, grid_shape, var_name):
    n_lev = grid_shape[0]
    v_3d = v_single.reshape(grid_shape)

    v_vert = (L_v @ v_3d.reshape(n_lev, -1)).reshape(grid_shape)
    delta_x_unfiltered = sigma_b * v_vert

    sig_h = SIGMA_HORIZONTAL.get(var_name, 2.5)
    delta_x_smoothed = np.zeros_like(delta_x_unfiltered)

    for k in range(n_lev):
        delta_x_smoothed[k] = gaussian_filter(delta_x_unfiltered[k], sigma=sig_h, mode="wrap")

    return delta_x_smoothed


def apply_B_half_transpose_var(r_single, L_v, sigma_b, grid_shape, var_name):
    n_lev = grid_shape[0]
    r_3d = r_single.reshape(grid_shape)

    sig_h = SIGMA_HORIZONTAL.get(var_name, 2.5)
    r_smoothed = np.zeros_like(r_3d)

    for k in range(n_lev):
        r_smoothed[k] = gaussian_filter(r_3d[k], sigma=sig_h, mode="wrap")

    r_scaled = sigma_b * r_smoothed
    r_flat = r_scaled.reshape(n_lev, -1)
    v_grad = (L_v.T @ r_flat).reshape(grid_shape)

    return v_grad.ravel()


def compute_weak_constraints(delta_x_dict, x_b_dict, lats, lons, heights):
    """Computes J_c penalty terms and incremental gradients for dynamic balances."""
    J_c = 0.0
    grad_c_dict = {var: np.zeros_like(delta_x_dict[var]) for var in VAR_LIST}

    # Grid spacings
    dlat = np.radians(np.abs(np.mean(np.diff(lats))))
    dlon = np.radians(np.abs(np.mean(np.diff(lons))))
    dy = EARTH_RADIUS * dlat

    lat_rad = np.radians(lats)
    f_coriolis = 2.0 * OMEGA_EARTH * np.sin(lat_rad)
    f_coriolis = np.where(np.abs(f_coriolis) < 1e-5, 1e-5, f_coriolis)  # Avoid equator div-by-zero

    lat_weight = np.sin(lat_rad)**2  # Tropics dampening factor
    lat_weight_3d = lat_weight[None, :, None]

    dp = delta_x_dict["p"] * 100.0  # Convert hPa to Pa
    du = delta_x_dict["u"]
    dv = delta_x_dict["v"]
    dT = delta_x_dict["t"]

    rho_approx = 1.225  # kg/m^3

    # --- 1. GEOSTROPHIC BALANCE PENALTY ---
    # dp/dy and dp/dx gradients
    dp_dy = np.gradient(dp, axis=1) / dy

    u_geo = -1.0 / (rho_approx * f_coriolis[None, :, None]) * dp_dy
    v_geo = np.zeros_like(u_geo)

    for j, lat_val in enumerate(lat_rad):
        dx_j = EARTH_RADIUS * np.cos(lat_val) * dlon
        if dx_j > 100.0:
            dp_dx_j = np.gradient(dp[:, j, :], axis=1) / dx_j
            v_geo[:, j, :] = 1.0 / (rho_approx * f_coriolis[j]) * dp_dx_j

    err_u = (du - u_geo) * lat_weight_3d
    err_v = (dv - v_geo) * lat_weight_3d

    J_geo = 0.5 * GAMMA_GEO * np.sum(err_u**2 + err_v**2)
    J_c += J_geo

    grad_c_dict["u"] += GAMMA_GEO * err_u
    grad_c_dict["v"] += GAMMA_GEO * err_v

    # --- 2. HYDROSTATIC BALANCE PENALTY ---
    dh = np.gradient(heights)
    dh_3d = dh[:, None, None]

    dp_dz = np.gradient(dp, axis=0) / dh_3d
    t_mean = np.maximum(x_b_dict["t"], 150.0)
    p_mean = np.maximum(x_b_dict["p"] * 100.0, 1000.0)

    dT_hydro = - (R_DRY * (t_mean**2) / (G_ACCEL * p_mean)) * dp_dz
    err_hydro = dT - dT_hydro

    J_hydro = 0.5 * GAMMA_HYDRO * np.sum(err_hydro**2)
    J_c += J_hydro

    grad_c_dict["t"] += GAMMA_HYDRO * err_hydro

    return J_c, grad_c_dict


# ==============================================================================
# 6. COST FUNCTION & GRADIENT EVALUATOR
# ==============================================================================
def cost_function_and_gradient(
    v_stacked, h_ops_dict, obs_dict, x_b_dict, L_v_dict, sigma_b_dict, grid_shape, heights, lats, lons
):
    n_grid = np.prod(grid_shape)
    J_b = 0.5 * np.sum(v_stacked**2)
    J_o = 0.0

    grad_stacked = np.copy(v_stacked)
    delta_x_dict = {}

    # Step 1: Forward transformation to state space
    for idx, var in enumerate(VAR_LIST):
        v_var = v_stacked[idx * n_grid : (idx + 1) * n_grid]
        delta_x_dict[var] = apply_B_half_var(v_var, L_v_dict[var], sigma_b_dict[var], grid_shape, var)

    # Step 2: Compute Observation Penalty (J_o)
    for idx, var in enumerate(VAR_LIST):
        v_var = v_stacked[idx * n_grid : (idx + 1) * n_grid]
        x_state = x_b_dict[var] + delta_x_dict[var]

        H_op = h_ops_dict[var]
        _, _, _, obs_y, obs_err = obs_dict[var]

        H_x = H_op.H(x_state)
        d = obs_y - H_x

        J_o += 0.5 * np.sum((d / obs_err)**2)

        obs_adj_input = -d / (obs_err**2)
        grid_adj_input = H_op.H_adjoint(obs_adj_input)
        grad_v_var = apply_B_half_transpose_var(
            grid_adj_input, L_v_dict[var], sigma_b_dict[var], grid_shape, var
        )

        grad_stacked[idx * n_grid : (idx + 1) * n_grid] += grad_v_var

    # Step 3: Compute Weak Balance Constraints Penalty (J_c)
    J_c, grad_c_dict = compute_weak_constraints(delta_x_dict, x_b_dict, lats, lons, heights)

    for idx, var in enumerate(VAR_LIST):
        grad_v_c = apply_B_half_transpose_var(
            grad_c_dict[var], L_v_dict[var], sigma_b_dict[var], grid_shape, var
        )
        grad_stacked[idx * n_grid : (idx + 1) * n_grid] += grad_v_c

    return J_b + J_o + J_c, grad_stacked


# ==============================================================================
# 7. MAIN DRIVER PIPELINE
# ==============================================================================
def main():
    args = parse_args()

    # Step 1: Load Background and Error Covariances
    x_b_dict, sigma_b_dict, L_v_dict, heights, lats, lons = load_gfs_and_bmatrix(
        args.gfs_file, args.bmatrix_file
    )
    grid_shape = x_b_dict["t"].shape
    n_grid = np.prod(grid_shape)

    # Step 2: Query NNJA-AI Catalog Observations
    obs_dict = fetch_nnja_observations(args.time, heights, lats, lons, x_b_dict)

    # Setup forward operators
    h_ops_dict = {}
    for var in VAR_LIST:
        obs_h, obs_lat, obs_lon, _, _ = obs_dict[var]
        h_ops_dict[var] = ObservationOperator3D(heights, lats, lons, obs_h, obs_lat, obs_lon)

    # Step 3: Run Constrained Minimization
    print("\n[4/5] Minimizing Cost Function J = J_b + J_o + J_c (Geostrophic + Hydrostatic)...")
    v0_stacked = np.zeros(5 * n_grid, dtype=np.float64)

    opt_result = minimize(
        fun=cost_function_and_gradient,
        x0=v0_stacked,
        args=(h_ops_dict, obs_dict, x_b_dict, L_v_dict, sigma_b_dict, grid_shape, heights, lats, lons),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": args.max_iter},
    )

    v_opt_stacked = opt_result.x
    print(f"\n3D-Var Convergence Finished in {opt_result.nit} iterations!")

    # Step 4: Reconstruct 5-Variable Analysis State
    x_a_dict = {}
    delta_x_dict = {}

    for idx, var in enumerate(VAR_LIST):
        v_opt_var = v_opt_stacked[idx * n_grid : (idx + 1) * n_grid]
        delta_x = apply_B_half_var(v_opt_var, L_v_dict[var], sigma_b_dict[var], grid_shape, var)
        
        # Enforce physical bounding for moisture
        if var == "q":
            analysis_val = np.maximum(x_b_dict[var] + delta_x, 1e-7)
            delta_x = analysis_val - x_b_dict[var]
        else:
            analysis_val = x_b_dict[var] + delta_x

        delta_x_dict[var] = delta_x
        x_a_dict[var] = analysis_val

        print(f"Var '{var.upper()}' | Max Abs Increment: {np.max(np.abs(delta_x)):.4e}")

    # Step 5: Export Constrained NetCDF Analysis Output
    print(f"\n[5/5] Exporting Constrained Analysis to: '{args.output}'")
    data_vars = {}

    for var in VAR_LIST:
        data_vars[f"{var}_background"] = (
            ["height", "latitude", "longitude"],
            x_b_dict[var].astype(np.float32),
            {"long_name": f"GFS Background State ({var.upper()})"},
        )
        data_vars[f"{var}_analysis"] = (
            ["height", "latitude", "longitude"],
            x_a_dict[var].astype(np.float32),
            {"long_name": f"Constrained 3D-Var Analysis ({var.upper()})"},
        )
        data_vars[f"{var}_increment"] = (
            ["height", "latitude", "longitude"],
            delta_x_dict[var].astype(np.float32),
            {"long_name": f"Constrained 3D-Var Increment ({var.upper()})"},
        )

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={"height": heights, "latitude": lats, "longitude": lons},
        attrs={
            "title": "5-Variable GFS 3D-Var Analysis with Geostrophic & Hydrostatic Constraints",
            "institution": "Anemoi DA Environment",
            "variables_assimilated": "p, t, u, v, q",
            "weak_constraints": f"Geostrophic (gamma={GAMMA_GEO}), Hydrostatic (gamma={GAMMA_HYDRO})",
            "iterations": str(opt_result.nit),
        },
    )

    ds_out.to_netcdf(
        args.output,
        encoding={v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars},
    )
    print("Pipeline Execution Completed Successfully.")


if __name__ == "__main__":
    main()
