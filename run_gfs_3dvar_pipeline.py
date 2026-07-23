#!/usr/bin/env python3
"""
Complete 3D-Var Data Assimilation Pipeline for GFS / Anemoi Environment
------------------------------------------------------------------------
Ingests:
  1. GFS Background State NetCDF (x_b)
  2. Generated B-Matrix Statistics NetCDF (Variances + Vertical Correlations)
  3. Observation NetCDF or Dictionary (y_obs)

Outputs:
  CF-Compliant NetCDF containing Background (x_b), Analysis (x_a), 
  and Analysis Increments (delta_x).
"""

import argparse
import sys
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cholesky
from scipy.optimize import minimize


# ==============================================================================
# 1. ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 3D-Var Data Assimilation on GFS Grid with precomputed B-Matrix."
    )
    parser.add_argument(
        "--gfs-file",
        type=str,
        required=True,
        help="Path to GFS NetCDF file containing background forecast state.",
    )
    parser.add_argument(
        "--bmatrix-file",
        type=str,
        required=True,
        help="Path to B-Matrix NetCDF file containing error variances and correlations.",
    )
    parser.add_argument(
        "--obs-file",
        type=str,
        default=None,
        help="Path to observation NetCDF file. If not provided, synthetic obs will be generated.",
    )
    parser.add_argument(
        "--var-name",
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
        "--max-iter",
        type=int,
        default=60,
        help="Maximum iterations for L-BFGS-B optimizer. Default: 60",
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

    # Ensure coordinates are strictly ascending for scipy interpolation
    lat_asc = np.all(np.diff(lats) > 0)
    if not lat_asc:
        # If latitude is descending, flip axis along latitude
        lats = lats[::-1]
        x_b = np.flip(x_b, axis=1)

    print(f"[2/5] Loading B-Matrix Error Statistics from: {bmatrix_path}")
    with xr.open_dataset(bmatrix_path) as ds_b:
        var_3d = ds_b[f"{var_name}_var"].values.astype(np.float64)
        if not lat_asc:
            var_3d = np.flip(var_3d, axis=1)
            
        sigma_b = np.sqrt(np.maximum(var_3d, 1e-8))  # Standard deviation 3D grid

        # Vertical correlation matrix & Cholesky decomposition
        vert_corr = ds_b[f"{var_name}_vert_corr"].values.astype(np.float64)
        # Add small jitter to diagonal for positive-definiteness
        vert_corr += np.eye(vert_corr.shape[0]) * 1e-6
        L_v = cholesky(vert_corr, lower=True)

    return x_b, sigma_b, L_v, heights, lats, lons


# ==============================================================================
# 3. OBSERVATION OPERATOR (TRILINEAR INTERPOLATION & ADJOINT)
# ==============================================================================
class ObservationOperator3D:
    """
    Handles 3D spatial trilinear interpolation H(x) and its adjoint H^T.
    """
    def __init__(self, heights, lats, lons, obs_heights, obs_lats, obs_lons):
        self.heights = heights
        self.lats = lats
        self.lons = lons
        self.obs_pts = np.column_stack((obs_heights, obs_lats, obs_lons))
        self.grid_shape = (len(heights), len(lats), len(lons))

    def H(self, grid_state):
        """Forward operator: Interpolates 3D grid state to observation points."""
        interp = RegularGridInterpolator(
            (self.heights, self.lats, self.lons),
            grid_state,
            bounds_error=False,
            fill_value=None,
        )
        return interp(self.obs_pts)

    def H_adjoint(self, obs_residuals):
        """
        Adjoint operator H^T: Maps observation-space residuals back onto the 3D grid.
        Uses a scatter-add scheme on neighboring grid nodes.
        """
        adj_grid = np.zeros(self.grid_shape, dtype=np.float64)
        
        # Determine surrounding cell indices for each observation
        for k_obs, (h, lat, lon) in enumerate(self.obs_pts):
            # Height index bounds
            i_h = np.searchsorted(self.heights, h) - 1
            i_h = np.clip(i_h, 0, len(self.heights) - 2)
            
            # Lat index bounds
            i_lat = np.searchsorted(self.lats, lat) - 1
            i_lat = np.clip(i_lat, 0, len(self.lats) - 2)
            
            # Lon index bounds
            i_lon = np.searchsorted(self.lons, lon) - 1
            i_lon = np.clip(i_lon, 0, len(self.lons) - 2)

            # Linear weights
            wh = (h - self.heights[i_h]) / (self.heights[i_h + 1] - self.heights[i_h])
            wlat = (lat - self.lats[i_lat]) / (self.lats[i_lat + 1] - self.lats[i_lat])
            wlon = (lon - self.lons[i_lon]) / (self.lons[i_lon + 1] - self.lons[i_lon])

            res = obs_residuals[k_obs]

            # Distribute residual to 8 corner points
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
# 4. CONTROL VARIABLE TRANSFORM & 3D-VAR COST FUNCTION
# ==============================================================================
def apply_B_half(v, L_v, sigma_b, grid_shape):
    """Calculates increment delta_x = diag(sigma_b) * (L_v x I_H) * v"""
    n_lev, n_lat, n_lon = grid_shape
    v_3d = v.reshape(grid_shape)
    
    # Vertical coupling transformation
    v_vert = (L_v @ v_3d.reshape(n_lev, -1)).reshape(grid_shape)
    
    # Pointwise scaling by 3D variance
    return sigma_b * v_vert


def apply_B_half_transpose(r, L_v, sigma_b, grid_shape):
    """Calculates adjoint transformation v = (B^{1/2})^T * r"""
    n_lev, n_lat, n_lon = grid_shape
    r_scaled = sigma_b * r.reshape(grid_shape)
    
    # Adjoint vertical coupling
    r_flat = r_scaled.reshape(n_lev, -1)
    v_grad = (L_v.T @ r_flat).reshape(grid_shape)
    
    return v_grad.ravel()


def cost_function_and_gradient(v, H_op, y_obs, sigma_o, x_b, L_v, sigma_b, grid_shape):
    """
    Computes Incremental Cost Function J(v) and Gradient grad_J(v):
      J(v) = 0.5 * v^T v + 0.5 * (H(x_b + B^{1/2}v) - y)^T R^{-1} (H(x_b + B^{1/2}v) - y)
    """
    # 1. Transform control variable v to increment delta_x
    delta_x = apply_B_half(v, L_v, sigma_b, grid_shape)
    x_state = x_b + delta_x

    # 2. Compute Innovation Vector: d = y - H(x)
    H_x = H_op.H(x_state)
    d = y_obs - H_x

    # 3. Cost terms
    J_b = 0.5 * np.sum(v**2)
    J_o = 0.5 * np.sum((d / sigma_o)**2)
    J_total = J_b + J_o

    # 4. Adjoint gradient computation
    obs_adj_input = -d / (sigma_o**2)
    grid_adj_input = H_op.H_adjoint(obs_adj_input)
    grad_v = v + apply_B_half_transpose(grid_adj_input, L_v, sigma_b, grid_shape)

    return J_total, grad_v


# ==============================================================================
# 5. SYNTHETIC OBSERVATION GENERATOR (FALLBACK)
# ==============================================================================
def generate_synthetic_observations(heights, lats, lons, x_b, n_obs=150):
    """Generates synthetic observations with noise for testing if obs file is omitted."""
    print(f"\n[3/5] Generating {n_obs} Synthetic Observations for pipeline validation...")
    np.random.seed(42)
    
    obs_h = np.random.uniform(heights.min(), heights.max(), n_obs)
    obs_lat = np.random.uniform(lats.min(), lats.max(), n_obs)
    obs_lon = np.random.uniform(lons.min(), lons.max(), n_obs)

    H_temp = ObservationOperator3D(heights, lats, lons, obs_h, obs_lat, obs_lon)
    true_b = H_temp.H(x_b)
    
    # Add synthetic warm/cool anomalies + Gaussian noise (error = 0.8 K)
    errors = np.full(n_obs, 0.8, dtype=np.float64)
    obs_values = true_b + np.random.normal(0.5, 0.8, n_obs)

    return obs_h, obs_lat, obs_lon, obs_values, errors


# ==============================================================================
# 6. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    args = parse_args()

    # Step 1: Load inputs
    x_b, sigma_b, L_v, heights, lats, lons = load_gfs_and_bmatrix(
        args.gfs_file, args.bmatrix_file, args.var_name
    )
    grid_shape = x_b.shape
    print(f"Domain Shape: Height={grid_shape[0]}, Lat={grid_shape[1]}, Lon={grid_shape[2]}")

    # Step 2: Ingest or Generate Observations
    if args.obs_file:
        print(f"\n[3/5] Ingesting Observations from NetCDF: {args.obs_file}")
        with xr.open_dataset(args.obs_file) as ds_obs:
            obs_h = ds_obs["height"].values
            obs_lat = ds_obs["latitude"].values
            obs_lon = ds_obs["longitude"].values
            obs_y = ds_obs["observation_value"].values
            obs_err = ds_obs["observation_error"].values
    else:
        obs_h, obs_lat, obs_lon, obs_y, obs_err = generate_synthetic_observations(
            heights, lats, lons, x_b
        )

    # Initialize 3D Spatial Observation Operator
    H_op = ObservationOperator3D(heights, lats, lons, obs_h, obs_lat, obs_lon)

    # Step 3: Minimize 3D-Var Cost Function
    print("\n[4/5] Running Incremental 3D-Var Minimization (L-BFGS-B)...")
    v0 = np.zeros(np.prod(grid_shape), dtype=np.float64)

    opt_result = minimize(
        fun=cost_function_and_gradient,
        x0=v0,
        args=(H_op, obs_y, obs_err, x_b, L_v, sigma_b, grid_shape),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": args.max_iter, "disp": True},
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

    ds_out.to_netcdf(args.output, encoding={v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars})
    print("Pipeline Execution Completed Successfully.")


if __name__ == "__main__":
    main()

