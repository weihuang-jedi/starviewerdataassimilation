#!/usr/bin/env python3
"""
Complete 3D-Var Data Assimilation Pipeline for GFS / Anemoi Environment
------------------------------------------------------------------------
Ingests:
  1. GFS Background State NetCDF (x_b)
  2. Generated B-Matrix Statistics NetCDF (Variances + Vertical Correlations)
  3. Real Observations via NNJA-AI Catalog (conv-adpupa-NC002001) or Fallback NetCDF

Outputs:
  CF-Compliant NetCDF containing Background (x_b), Analysis (x_a),
  and Analysis Increments (delta_x).
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

# Suppress minor catalog warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nnja_ai")


# ==============================================================================
# 1. ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 3D-Var Data Assimilation on GFS Grid with NNJA-AI observations."
    )
    parser.add_argument(
        "--gfs_file",
        type=str,
        required=True,
        help="Path to GFS NetCDF file containing background forecast state.",
    )
    parser.add_argument(
        "--bmatrix_file",
        type=str,
        required=True,
        help="Path to B-Matrix NetCDF file containing error variances and correlations.",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="2023-07-01",
        help="Assimilation date/time string (e.g. '2023-07-01' or '2023-07-01T06:00:00').",
    )
    parser.add_argument(
        "--obs_file",
        type=str,
        default=None,
        help="Optional path to local observation NetCDF. If omitted, NNJA-AI catalog is queried.",
    )
    parser.add_argument(
        "--var_name",
        type=str,
        default="t",
        help="Variable name to assimilate (e.g., 't', 'u', 'v', 'q'). Default: 't'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gfs_3dvar_analysis.nc",
        help="Output NetCDF file name. Default: 'gfs_3dvar_analysis.nc'",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=30,
        help="Maximum iterations for L-BFGS-B optimizer. Default: 30",
    )
    return parser.parse_args()


# ==============================================================================
# 2. DATA INGESTION & B-MATRIX LOADING
# ==============================================================================
def load_gfs_and_bmatrix(gfs_path: str, bmatrix_path: str, var_name: str):
    """Loads 3D background state, grid coordinates, and B-matrix statistics."""
    print(f"[1/5] Loading Background State from: {gfs_path}")
    with xr.open_dataset(gfs_path) as ds_gfs:
        x_b = ds_gfs[var_name].values.astype(np.float64)  # (height, lat, lon)
        heights = ds_gfs["height"].values.astype(np.float64)
        lats = ds_gfs["latitude"].values.astype(np.float64)
        lons = ds_gfs["longitude"].values.astype(np.float64)

    # Ensure latitude coordinates are strictly ascending
    lat_asc = np.all(np.diff(lats) > 0)
    if not lat_asc:
        lats = lats[::-1]
        x_b = np.flip(x_b, axis=1)

    print(f"[2/5] Loading B-Matrix Error Statistics from: {bmatrix_path}")
    with xr.open_dataset(bmatrix_path) as ds_b:
        var_3d = ds_b[f"{var_name}_var"].values.astype(np.float64)
        if not lat_asc:
            var_3d = np.flip(var_3d, axis=1)

        sigma_b = np.sqrt(np.maximum(var_3d, 1e-8))

        vert_corr = ds_b[f"{var_name}_vert_corr"].values.astype(np.float64)
        vert_corr += np.eye(vert_corr.shape[0]) * 1e-6
        L_v = cholesky(vert_corr, lower=True)

    return x_b, sigma_b, L_v, heights, lats, lons


# ==============================================================================
# 3. NNJA-AI OBSERVATION FETCHING & CLEANING
# ==============================================================================
def fetch_nnja_observations(time_str: str, heights, lats, lons, x_b):
    """Fetches real radiosonde observations from NNJA-AI catalog."""
    if not HAS_NNJA:
        print("Warning: 'nnja_ai' library not installed. Falling back to synthetic obs.")
        return generate_synthetic_observations(heights, lats, lons, x_b)

    # Extract date part (YYYY-MM-DD) for NNJA-AI dataset manifest indexing
    date_part = time_str.split("T")[0]
    print(f"\n[3/5] Querying NNJA-AI Catalog ('conv-adpupa-NC002001') for date: {date_part}...")

    try:
        catalog = DataCatalog(mirror="gcp_brightband")
        ds_upr = catalog["conv-adpupa-NC002001"]
        subset = ds_upr.sel(time=date_part)  # Select by date string
        df_upr = subset.load_dataset(backend="pandas")
    except Exception as e:
        print(f"Error accessing NNJA-AI catalog: {e}. Falling back to synthetic obs.")
        return generate_synthetic_observations(heights, lats, lons, x_b)

    if df_upr.empty:
        print("Warning: Retrieved empty observation dataset. Falling back to synthetic obs.")
        return generate_synthetic_observations(heights, lats, lons, x_b)

    # Normalize Longitudes to match GFS grid domain
    grid_lon_min, grid_lon_max = lons.min(), lons.max()
    if grid_lon_min < 0:
        df_upr["LON"] = np.where(df_upr["LON"] > 180, df_upr["LON"] - 360, df_upr["LON"])
    else:
        df_upr["LON"] = np.where(df_upr["LON"] < 0, df_upr["LON"] + 360, df_upr["LON"])

    temp_cols = [c for c in df_upr.columns if "TMDB" in c or "TMP" in c]

    obs_records = []
    lat_min, lat_max = lats.min(), lats.max()
    h_min, h_max = heights.min(), heights.max()

    for _, row in df_upr.iterrows():
        lat, lon = row["LAT"], row["LON"]
        if not (lat_min <= lat <= lat_max and grid_lon_min <= lon <= grid_lon_max):
            continue

        for col in temp_cols:
            val = row[col]
            if pd.notna(val) and -100.0 < val < 350.0:
                p_level = 850.0
                for p in [1000, 925, 850, 700, 500, 300, 250, 200, 100]:
                    if str(p) in col:
                        p_level = float(p)
                        break

                # Map pressure level to height in meters
                h_approx = 44330.0 * (1.0 - (p_level / 1013.25)**0.1903)
                h_clamped = np.clip(h_approx, h_min, h_max)

                if val < 150.0:
                    val += 273.15

                obs_records.append({
                    "height": h_clamped,
                    "latitude": lat,
                    "longitude": lon,
                    "observation_value": val,
                    "observation_error": 1.0
                })

    obs_df = pd.DataFrame(obs_records)

    if obs_df.empty:
        print("Warning: No observations found inside domain. Falling back to synthetic obs.")
        return generate_synthetic_observations(heights, lats, lons, x_b)

    print(f"Successfully ingested {len(obs_df)} observation points from NNJA-AI.")

    return (
        obs_df["height"].values,
        obs_df["latitude"].values,
        obs_df["longitude"].values,
        obs_df["observation_value"].values,
        obs_df["observation_error"].values,
    )


def generate_synthetic_observations(heights, lats, lons, x_b, n_obs=150):
    """Realistic synthetic observation generator sampled from background state."""
    print(f"Generating {n_obs} synthetic observations anchored to x_b...")
    np.random.seed(42)
    obs_h = np.random.uniform(heights.min(), heights.max(), n_obs)
    obs_lat = np.random.uniform(lats.min(), lats.max(), n_obs)
    obs_lon = np.random.uniform(lons.min(), lons.max(), n_obs)

    H_temp = ObservationOperator3D(heights, lats, lons, obs_h, obs_lat, obs_lon)
    x_b_obs = H_temp.H(x_b)

    # Add realistic +0.5 K bias and 1.0 K Gaussian error
    obs_values = x_b_obs + np.random.normal(0.5, 1.0, n_obs)
    errors = np.full(n_obs, 1.0, dtype=np.float64)
    return obs_h, obs_lat, obs_lon, obs_values, errors


# ==============================================================================
# 4. OBSERVATION OPERATOR (TRILINEAR INTERPOLATION & ADJOINT)
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
# 5. CONTROL VARIABLE TRANSFORM & 3D-VAR COST FUNCTION
# ==============================================================================
def apply_B_half(v, L_v, sigma_b, grid_shape):
    n_lev = grid_shape[0]
    v_3d = v.reshape(grid_shape)
    v_vert = (L_v @ v_3d.reshape(n_lev, -1)).reshape(grid_shape)
    return sigma_b * v_vert


def apply_B_half_transpose(r, L_v, sigma_b, grid_shape):
    n_lev = grid_shape[0]
    r_scaled = sigma_b * r.reshape(grid_shape)
    r_flat = r_scaled.reshape(n_lev, -1)
    v_grad = (L_v.T @ r_flat).reshape(grid_shape)
    return v_grad.ravel()


def cost_function_and_gradient(v, H_op, y_obs, sigma_o, x_b, L_v, sigma_b, grid_shape):
    delta_x = apply_B_half(v, L_v, sigma_b, grid_shape)
    x_state = x_b + delta_x

    H_x = H_op.H(x_state)
    d = y_obs - H_x

    J_b = 0.5 * np.sum(v**2)
    J_o = 0.5 * np.sum((d / sigma_o)**2)
    J_total = J_b + J_o

    obs_adj_input = -d / (sigma_o**2)
    grid_adj_input = H_op.H_adjoint(obs_adj_input)
    grad_v = v + apply_B_half_transpose(grid_adj_input, L_v, sigma_b, grid_shape)

    return J_total, grad_v


# ==============================================================================
# 6. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    args = parse_args()

    # Step 1: Load GFS background and B-matrix
    x_b, sigma_b, L_v, heights, lats, lons = load_gfs_and_bmatrix(
        args.gfs_file, args.bmatrix_file, args.var_name
    )
    grid_shape = x_b.shape
    print(f"Domain Shape: Height={grid_shape[0]}, Lat={grid_shape[1]}, Lon={grid_shape[2]}")

    # Step 2: Fetch Observations
    if args.obs_file:
        print(f"\n[3/5] Ingesting Observations from NetCDF File: {args.obs_file}")
        with xr.open_dataset(args.obs_file) as ds_obs:
            obs_h = ds_obs["height"].values
            obs_lat = ds_obs["latitude"].values
            obs_lon = ds_obs["longitude"].values
            obs_y = ds_obs["observation_value"].values
            obs_err = ds_obs["observation_error"].values
    else:
        obs_h, obs_lat, obs_lon, obs_y, obs_err = fetch_nnja_observations(
            args.time, heights, lats, lons, x_b
        )

    # Initialize Forward Operator
    H_op = ObservationOperator3D(heights, lats, lons, obs_h, obs_lat, obs_lon)

    # Step 3: Run 3D-Var Minimization
    print("\n[4/5] Running Incremental 3D-Var Minimization (L-BFGS-B)...")
    v0 = np.zeros(np.prod(grid_shape), dtype=np.float64)

    # Note: 'disp' removed from options dict to prevent SciPy warning
    opt_result = minimize(
        fun=cost_function_and_gradient,
        x0=v0,
        args=(H_op, obs_y, obs_err, x_b, L_v, sigma_b, grid_shape),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": args.max_iter},
    )

    # Step 4: Reconstruct Analysis State (x_a = x_b + delta_x)
    v_opt = opt_result.x
    delta_x_opt = apply_B_half(v_opt, L_v, sigma_b, grid_shape)
    x_a = x_b + delta_x_opt

    print(f"\n3D-Var Convergence Finished!")
    print(f"Max Absolute Increment: {np.max(np.abs(delta_x_opt)):.4f} K")
    print(f"Mean Increment: {np.mean(delta_x_opt):.4f} K")

    # Step 5: Export Results to NetCDF
    print(f"\n[5/5] Exporting Analysis Results to NetCDF: '{args.output}'")
    ds_out = xr.Dataset(
        data_vars={
            f"{args.var_name}_background": (
                ["height", "latitude", "longitude"],
                x_b.astype(np.float32),
                {"long_name": "GFS Background Forecast State"},
            ),
            f"{args.var_name}_analysis": (
                ["height", "latitude", "longitude"],
                x_a.astype(np.float32),
                {"long_name": "3D-Var Analysis State"},
            ),
            f"{args.var_name}_increment": (
                ["height", "latitude", "longitude"],
                delta_x_opt.astype(np.float32),
                {"long_name": "3D-Var Analysis Increment (x_a - x_b)"},
            ),
        },
        coords={
            "height": heights,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": "GFS 3D-Var Analysis Output",
            "institution": "Anemoi DA Environment",
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
