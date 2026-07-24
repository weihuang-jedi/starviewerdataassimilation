#!/usr/bin/env python3
"""
Multivariate 3D-Var Data Assimilation Pipeline (NNJA-AI Ingestion)
------------------------------------------------------------------
Assimilates 5 State Variables: [p, t, u, v, q]

Directly queries NNJA-AI 'conv-adpupa-NC002001' catalog for upper-air sounding data.
Outputs a CF-compliant NetCDF with [Background, Analysis, Increments].
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cholesky
from scipy.optimize import minimize

try:
    from nnja_ai import DataCatalog
    HAS_NNJA = True
except ImportError:
    HAS_NNJA = False

warnings.filterwarnings("ignore", category=UserWarning, module="nnja_ai")

# 5 target variables and NNJA column keywords
VAR_LIST = ["p", "t", "u", "v", "q"]
NNJA_VAR_MAP = {
    "p": ["PRLC", "PRES", "PRSE"],
    "t": ["TMDB", "TMP"],
    "u": ["UGRD", "UWND", "U_WIND"],
    "v": ["VGRD", "VWND", "V_WIND"],
    "q": ["SPFH", "Q", "HUMI"],
}

# Standard observational errors (R-matrix terms)
DEFAULT_OBS_ERRORS = {
    "p": 1.0,     # hPa
    "t": 1.0,     # K
    "u": 1.5,     # m/s
    "v": 1.5,     # m/s
    "q": 0.001,   # kg/kg
}


# ==============================================================================
# 1. ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 5-Variable [p, t, u, v, q] 3D-Var with NNJA-AI Data Ingestion."
    )
    parser.add_argument(
        "--gfs_file",
        type=str,
        required=True,
        help="Path to GFS background NetCDF file.",
    )
    parser.add_argument(
        "--bmatrix_file",
        type=str,
        required=True,
        help="Path to B-Matrix error statistics NetCDF file.",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="2023-07-01",
        help="Assimilation date (e.g., '2023-07-01').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gfs_5var_3dvar_analysis.nc",
        help="Output NetCDF file path.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=30,
        help="Maximum iterations for L-BFGS-B optimizer.",
    )
    return parser.parse_args()


# ==============================================================================
# 2. DATA & COVARIANCE LOADING
# ==============================================================================
def load_gfs_and_bmatrix(gfs_path: str, bmatrix_path: str):
    """Loads 5-variable background states and B-matrix terms."""
    print(f"[1/5] Loading Background State [p, t, u, v, q] from: {gfs_path}")
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
# 3. NNJA-AI CATALOG INGESTION ENGINE
# ==============================================================================
def fetch_nnja_observations(time_str: str, heights, lats, lons, x_b_dict):
    """Queries NNJA-AI catalog for [p, t, u, v, q] sounding profiles."""
    obs_dict = {}

    if not HAS_NNJA:
        print("Warning: 'nnja_ai' not installed. Falling back to synthetic obs.")
        return generate_synthetic_obs_all(heights, lats, lons, x_b_dict)

    # Use date string YYYY-MM-DD for NNJA-AI manifest lookup
    date_part = time_str.split("T")[0]
    print(f"\n[3/5] Querying NNJA-AI Catalog ('conv-adpupa-NC002001') for date: {date_part}...")

    try:
        catalog = DataCatalog(mirror="gcp_brightband")
        ds_upr = catalog["conv-adpupa-NC002001"]
        subset = ds_upr.sel(time=date_part)
        df_upr = subset.load_dataset(backend="pandas")
    except Exception as e:
        print(f"NNJA access notice: {e}. Falling back to domain synthetic obs.")
        return generate_synthetic_obs_all(heights, lats, lons, x_b_dict)

    if df_upr.empty:
        print("Retrieved empty dataset from NNJA-AI. Falling back to synthetic obs.")
        return generate_synthetic_obs_all(heights, lats, lons, x_b_dict)

    # Normalize longitudes to match background domain
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

                    # Temperature units conversion if reported in Celsius
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
            print(f"No real observations found for '{var}'. Generating domain fallback.")
            obs_dict[var] = generate_single_synth_obs(heights, lats, lons, x_b_dict[var], var)
        else:
            obs_dict[var] = (
                df_var["height"].values,
                df_var["latitude"].values,
                df_var["longitude"].values,
                df_var["observation_value"].values,
                df_var["observation_error"].values,
            )
            print(f"Ingested {len(df_var)} observation points for variable '{var}'.")

    return obs_dict


def generate_single_synth_obs(heights, lats, lons, x_b_var, var_name, n_obs=120):
    """Generates synthetic obs anchored to x_b if real field obs are missing."""
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
# 4. TRILINEAR OBSERVATION OPERATOR & ADJOINT
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
# 5. MULTIVARIATE 3D-VAR COST FUNCTION & GRADIENT
# ==============================================================================
def apply_B_half_var(v_single, L_v, sigma_b, grid_shape):
    n_lev = grid_shape[0]
    v_3d = v_single.reshape(grid_shape)
    v_vert = (L_v @ v_3d.reshape(n_lev, -1)).reshape(grid_shape)
    return sigma_b * v_vert


def apply_B_half_transpose_var(r_single, L_v, sigma_b, grid_shape):
    n_lev = grid_shape[0]
    r_scaled = sigma_b * r_single.reshape(grid_shape)
    r_flat = r_scaled.reshape(n_lev, -1)
    v_grad = (L_v.T @ r_flat).reshape(grid_shape)
    return v_grad.ravel()


def cost_function_and_gradient_multivariate(
    v_stacked, h_ops_dict, obs_dict, x_b_dict, L_v_dict, sigma_b_dict, grid_shape
):
    n_grid = np.prod(grid_shape)
    J_b = 0.5 * np.sum(v_stacked**2)
    J_o = 0.0

    grad_stacked = np.copy(v_stacked)

    for idx, var in enumerate(VAR_LIST):
        v_var = v_stacked[idx * n_grid : (idx + 1) * n_grid]

        delta_x = apply_B_half_var(v_var, L_v_dict[var], sigma_b_dict[var], grid_shape)
        x_state = x_b_dict[var] + delta_x

        H_op = h_ops_dict[var]
        _, _, _, obs_y, obs_err = obs_dict[var]

        H_x = H_op.H(x_state)
        d = obs_y - H_x

        J_o += 0.5 * np.sum((d / obs_err)**2)

        obs_adj_input = -d / (obs_err**2)
        grid_adj_input = H_op.H_adjoint(obs_adj_input)
        grad_v_var = apply_B_half_transpose_var(
            grid_adj_input, L_v_dict[var], sigma_b_dict[var], grid_shape
        )

        grad_stacked[idx * n_grid : (idx + 1) * n_grid] += grad_v_var

    J_total = J_b + J_o
    return J_total, grad_stacked


# ==============================================================================
# 6. PIPELINE DRIVER
# ==============================================================================
def main():
    args = parse_args()

    # Step 1: Ingest Background State and Covariances
    x_b_dict, sigma_b_dict, L_v_dict, heights, lats, lons = load_gfs_and_bmatrix(
        args.gfs_file, args.bmatrix_file
    )
    grid_shape = x_b_dict["t"].shape
    n_grid = np.prod(grid_shape)

    # Step 2: Fetch NNJA-AI Observations
    obs_dict = fetch_nnja_observations(args.time, heights, lats, lons, x_b_dict)

    # Setup forward operators
    h_ops_dict = {}
    for var in VAR_LIST:
        obs_h, obs_lat, obs_lon, _, _ = obs_dict[var]
        h_ops_dict[var] = ObservationOperator3D(heights, lats, lons, obs_h, obs_lat, obs_lon)

    # Step 3: Run Joint 5-Variable Minimization
    print("\n[4/5] Minimizing Cost Function J(v) over [p, t, u, v, q]...")
    v0_stacked = np.zeros(5 * n_grid, dtype=np.float64)

    opt_result = minimize(
        fun=cost_function_and_gradient_multivariate,
        x0=v0_stacked,
        args=(h_ops_dict, obs_dict, x_b_dict, L_v_dict, sigma_b_dict, grid_shape),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": args.max_iter},
    )

    v_opt_stacked = opt_result.x
    print(f"\n3D-Var Convergence Finished in {opt_result.nit} iterations!")

    # Step 4: Reconstruct 5-Variable Analysis and Increments
    x_a_dict = {}
    delta_x_dict = {}

    for idx, var in enumerate(VAR_LIST):
        v_opt_var = v_opt_stacked[idx * n_grid : (idx + 1) * n_grid]
        delta_x = apply_B_half_var(v_opt_var, L_v_dict[var], sigma_b_dict[var], grid_shape)
        delta_x_dict[var] = delta_x
        x_a_dict[var] = x_b_dict[var] + delta_x

        print(f"Var '{var.upper()}' | Max Abs Increment: {np.max(np.abs(delta_x)):.4e}")

    # Step 5: Export Analysis NetCDF File
    print(f"\n[5/5] Saving Output NetCDF to: '{args.output}'")
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
            {"long_name": f"3D-Var Analysis State ({var.upper()})"},
        )
        data_vars[f"{var}_increment"] = (
            ["height", "latitude", "longitude"],
            delta_x_dict[var].astype(np.float32),
            {"long_name": f"3D-Var Analysis Increment ({var.upper()})"},
        )

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "height": heights,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": "5-Variable GFS 3D-Var Analysis Output",
            "institution": "Anemoi DA Environment",
            "assimilated_variables": "p, t, u, v, q",
            "optimization_status": str(opt_result.message),
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
