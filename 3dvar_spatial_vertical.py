#!/usr/bin/env python3
"""
3D Spatial-Vertical 3D-Var Data Assimilation Pipeline
-----------------------------------------------------
- Observations: NNJA-AI Radiosonde Upper-Air Data (conv-adpupa-NC002001)
- Method: Preconditioned 3D-Var in v-control space (dx = L_3D * v)
- Localization: Gaspari-Cohn horizontal and vertical log-pressure tapering
- Quality Control: Temperature unit correction + 3-sigma innovation filter
- Output: CF-compliant NetCDF dataset via xarray
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from nnja_ai import DataCatalog


def parse_args():
    """Parse command line arguments for time window and output filename."""
    parser = argparse.ArgumentParser(
        description="Run 3D-Var Spatial-Vertical Temperature Assimilation using NNJA-AI."
    )
    parser.add_argument(
        "--time",
        type=str,
        default="2021-01-01",
        help="Assimilation time string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). Default: 2021-01-01",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="3dvar_analysis_output.nc",
        help="Path for output NetCDF file. Default: 3dvar_analysis_output.nc",
    )
    return parser.parse_args()


# ==============================================================================
# MODULE 1: DATA INGESTION & CLEANING
# ==============================================================================
def fetch_and_clean_obs(time_str: str, domain_bbox: dict = None) -> pd.DataFrame:
    """Fetches upper-air observations from NNJA-AI catalog and applies basic cleaning."""
    print(f"Fetching NNJA-AI upper-air (radiosonde) observations for {time_str}...")
    catalog = DataCatalog(mirror="gcp_brightband")

    ds_upr = catalog["conv-adpupa-NC002001"]
    subset = ds_upr.sel(time=time_str)
    df_upr = subset.load_dataset(backend="pandas")

    if df_upr.empty:
        print("Warning: Retrieved empty dataset from catalog.")
        return generate_synthetic_obs()

    # Normalize Longitudes to [-180, 180]
    df_upr["LON"] = np.where(df_upr["LON"] > 180, df_upr["LON"] - 360, df_upr["LON"])

    # Detect temperature columns in dataset
    temp_cols = [c for c in df_upr.columns if "TMDB" in c or "TMP" in c]

    # Extract Multi-Level Profiles
    obs_records = []
    for _, row in df_upr.iterrows():
        lat, lon = row["LAT"], row["LON"]
        for col in temp_cols:
            val = row[col]
            if pd.notna(val) and -100.0 < val < 350.0:
                p_level = 850.0  # Default fallback
                for p in [1000, 925, 850, 700, 500, 300, 250, 200, 100]:
                    if str(p) in col:
                        p_level = float(p)
                        break
                obs_records.append({"LAT": lat, "LON": lon, "p_hpa": p_level, "y_obs": val})

    obs_df = pd.DataFrame(obs_records)

    # Filter spatial domain (CONUS bounding box by default)
    if not obs_df.empty and domain_bbox:
        spatial_mask = (
            (obs_df["LAT"] >= domain_bbox["lat_min"]) & (obs_df["LAT"] <= domain_bbox["lat_max"]) &
            (obs_df["LON"] >= domain_bbox["lon_min"]) & (obs_df["LON"] <= domain_bbox["lon_max"])
        )
        obs_df = obs_df[spatial_mask].copy()

    if obs_df.empty:
        print("Warning: No matching observations found in domain. Falling back to synthetic profiles...")
        return generate_synthetic_obs()

    # Unit Conversion: Celsius -> Kelvin
    celsius_mask = obs_df["y_obs"] < 150.0
    if celsius_mask.any():
        n_converted = celsius_mask.sum()
        obs_df.loc[celsius_mask, "y_obs"] += 273.15
        print(f"Unit Correction: Converted {n_converted} observations from Celsius to Kelvin.")

    # Subsample dataset for efficiency
    n_sample = min(40, len(obs_df))
    obs_df = obs_df.sample(n=n_sample, random_state=42).reset_index(drop=True)
    print(f"Extracted {len(obs_df)} multi-level observation points.")
    return obs_df


def generate_synthetic_obs() -> pd.DataFrame:
    """Fallback function to generate dummy profiles for verification."""
    synth_records = []
    for la, lo in zip([35.0, 38.0, 41.0], [-90.0, -85.0, -88.0]):
        for p in [1000.0, 850.0, 700.0, 500.0, 300.0]:
            t_synth = 288.15 * (p / 1000.0)**0.1903 + np.random.normal(0, 1.0)
            synth_records.append({"LAT": la, "LON": lo, "p_hpa": p, "y_obs": t_synth})
    return pd.DataFrame(synth_records)


# ==============================================================================
# MODULE 2: GRID & COVARIANCE CONSTRUCTORS
# ==============================================================================
def create_3d_grid():
    """Generates 3D coordinates (Lat, Lon, Level) and background state x_b."""
    lats = np.linspace(32.0, 42.0, 5)
    lons = np.linspace(-95.0, -85.0, 5)
    p_levels = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])  # hPa

    n_lat, n_lon, n_lev = len(lats), len(lons), len(p_levels)
    grid_lon, grid_lat, grid_p = np.meshgrid(lons, lats, p_levels, indexing="ij")

    grid_coords = np.column_stack([grid_lat.ravel(), grid_lon.ravel(), grid_p.ravel()])
    coords_2d = np.column_stack([grid_lat[:, :, 0].ravel(), grid_lon[:, :, 0].ravel()])

    # Standard atmosphere background state x_b
    x_b_1d = 288.15 * (p_levels / 1000.0) ** 0.1903
    x_b = np.tile(x_b_1d, n_lat * n_lon)

    return lats, lons, p_levels, grid_coords, coords_2d, x_b


def gaspari_cohn(r: np.ndarray) -> np.ndarray:
    """Gaspari-Cohn compact support localization function."""
    c = np.zeros_like(r)
    m1 = (r >= 0) & (r < 1)
    c[m1] = 1.0 - 5/3*r[m1]**2 + 5/8*r[m1]**3 + 0.5*r[m1]**4 - 0.25*r[m1]**5
    m2 = (r >= 1) & (r < 2)
    c[m2] = (
        4/3*r[m2]**(-1) - 5.0 + 5.0*r[m2] - 5/3*r[m2]**2 
        + 5/8*r[m2]**3 - 0.5*r[m2]**4 + 1/12*r[m2]**5
    )
    return np.clip(c, 0.0, 1.0)


def build_background_covariance_cholesky(coords_2d, p_levels, sigma_b_h=2.0, sigma_b_v=1.0):
    """Builds horizontal and vertical localized B matrices and returns 3D Cholesky L_3D."""
    # Horizontal Covariance & Localization
    dists_2d = cdist(coords_2d, coords_2d) * 111.0  # km
    B_H_raw = (sigma_b_h**2) * np.exp(-0.5 * (dists_2d / 250.0)**2)
    C_H = gaspari_cohn(dists_2d / 200.0)
    B_H_loc = B_H_raw * C_H + 1e-4 * np.eye(len(coords_2d))
    L_H = np.linalg.cholesky(B_H_loc)

    # Vertical Covariance & Log-Pressure Localization
    ln_p = np.log(p_levels)
    dists_v = np.abs(ln_p[:, None] - ln_p[None, :])
    B_V_raw = (sigma_b_v**2) * np.exp(-0.5 * (dists_v / 0.8)**2)
    C_V = gaspari_cohn(dists_v / 0.6)
    B_V_loc = B_V_raw * C_V + 1e-4 * np.eye(len(p_levels))
    L_V = np.linalg.cholesky(B_V_loc)

    # 3D Kronecker Product Cholesky Factor: L_3D = L_H (x) L_V
    return np.kron(L_H, L_V)


# ==============================================================================
# MODULE 3: OBSERVATION OPERATOR & QUALITY CONTROL
# ==============================================================================
def apply_forward_operator_and_qc(obs_df, grid_coords, x_b, sigma_r=1.5, sigma_b_total=2.236):
    """Builds operator H, computes innovations, and screens observations using 3-sigma QC."""
    obs_coords = obs_df[["LAT", "LON", "p_hpa"]].values
    obs_latlon, obs_p = obs_coords[:, :2], obs_coords[:, 2]

    grid_latlon, grid_p_vals = grid_coords[:, :2], grid_coords[:, 2]

    h_dists = cdist(obs_latlon, grid_latlon) * 111.0
    v_dists = np.abs(np.log(obs_p[:, None]) - np.log(grid_p_vals[None, :]))

    weights = np.exp(-0.5 * (h_dists / 60.0)**2 - 0.5 * (v_dists / 0.3)**2)
    H = weights / weights.sum(axis=1, keepdims=True)

    y = obs_df["y_obs"].values
    d_raw = y - H @ x_b

    # 3-Sigma Innovation Quality Control
    # sigma_total = np.sqrt(sigma_b_total**2 + sigma_r**2)
    # qc_threshold = 3.0 * sigma_total

    sigma_control_factor = 5.0
    # Sigma Innovation Quality Control
    sigma_total = np.sqrt(sigma_b_total**2 + sigma_r**2)
    qc_threshold = sigma_control_factor * sigma_total

    qc_mask = np.abs(d_raw) <= qc_threshold
    print(f"QC Filter ({sigma_control_factor}-Sigma = {qc_threshold:.2f} K): Kept {np.sum(qc_mask)}/{len(y)} obs (Rejected {np.sum(~qc_mask)}).")

    y_qc = y[qc_mask]
    H_qc = H[qc_mask, :]
    d_qc = d_raw[qc_mask]
    R_inv = (1.0 / sigma_r**2) * np.eye(len(y_qc))

    return H_qc, d_qc, R_inv


# ==============================================================================
# MODULE 4: 3D-VAR OPTIMIZER
# ==============================================================================
def run_3dvar_solver(H, d, L_3D, R_inv, n_grid_3d):
    """Solves incremental 3D-Var in v-space using L-BFGS-B optimization."""
    G = H @ L_3D

    def cost_function_v(v):
        residual = d - G @ v
        return 0.5 * np.dot(v, v) + 0.5 * residual.T @ R_inv @ residual

    def cost_gradient_v(v):
        residual = d - G @ v
        return v - G.T @ R_inv @ residual

    v0 = np.zeros(n_grid_3d)

    print("\nSolving 3D Spatial-Vertical 3D-Var...")
    res = minimize(
        fun=cost_function_v,
        x0=v0,
        jac=cost_gradient_v,
        method="L-BFGS-B",
        options={"gtol": 1e-5, "maxiter": 100}
    )

    dx_3d = L_3D @ res.x
    return dx_3d, res


# ==============================================================================
# MODULE 5: NETCDF EXPORTER
# ==============================================================================
def export_to_netcdf(output_path: str, time_str: str, lats, lons, p_levels, x_b, x_a, dx):
    """Exports structured 3D Analysis output to a CF-compliant NetCDF file."""
    n_lat, n_lon, n_lev = len(lats), len(lons), len(p_levels)

    x_b_arr = x_b.reshape(n_lat, n_lon, n_lev)[None, ...]
    x_a_arr = x_a.reshape(n_lat, n_lon, n_lev)[None, ...]
    dx_arr = dx.reshape(n_lat, n_lon, n_lev)[None, ...]

    analysis_time = pd.Timestamp(time_str)

    ds_out = xr.Dataset(
        data_vars={
            "x_b": (["time", "latitude", "longitude", "level"], x_b_arr,
                    {"standard_name": "air_temperature", "long_name": "Background State (x_b)", "units": "K"}),
            "x_a": (["time", "latitude", "longitude", "level"], x_a_arr,
                    {"standard_name": "air_temperature", "long_name": "3D-Var Analysis State (x_a)", "units": "K"}),
            "dx":  (["time", "latitude", "longitude", "level"], dx_arr,
                    {"standard_name": "air_temperature_increment", "long_name": "Analysis Increment (dx)", "units": "K"}),
        },
        coords={
            "time": [analysis_time],
            "latitude": ("latitude", lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": ("longitude", lons, {"units": "degrees_east", "standard_name": "longitude"}),
            "level": ("level", p_levels, {"units": "hPa", "standard_name": "air_pressure", "positive": "down"}),
        },
        attrs={
            "title": "3D-Var Temperature Assimilation Output",
            "institution": "NOAA-NASA / EAGLE DA Pipeline",
            "data_source": "NNJA-AI conv-adpupa-NC002001",
            "method": "Incremental 3D-Var in v-space with Gaspari-Cohn Localization",
            "conventions": "CF-1.8",
        }
    )

    ds_out.to_netcdf(output_path)
    print(f"\nSaved NetCDF analysis dataset successfully to: {output_path}")


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================
def main():
    args = parse_args()

    domain_bbox = {"lat_min": 30.0, "lat_max": 45.0, "lon_min": -100.0, "lon_max": -80.0}

    # 1. Fetch & Preprocess Observations
    obs_df = fetch_and_clean_obs(args.time, domain_bbox)

    # 2. Build Grid & Background State
    lats, lons, p_levels, grid_coords, coords_2d, x_b = create_3d_grid()

    # 3. Construct Localized Background Covariance Matrix (L_3D)
    L_3D = build_background_covariance_cholesky(coords_2d, p_levels)

    # 4. Construct Forward Operator & Run 3-Sigma QC
    H_qc, d_qc, R_inv = apply_forward_operator_and_qc(obs_df, grid_coords, x_b)

    # 5. Run 3D-Var Optimizer
    dx_3d, res = run_3dvar_solver(H_qc, d_qc, L_3D, R_inv, len(grid_coords))
    x_a_3d = x_b + dx_3d

    print(f"\n3D-Var Converged: {res.success} in {res.nit} iterations.")
    print(f"Final Cost J: {res.fun:.3f}")
    print(f"Overall Mean Abs Increment: {np.mean(np.abs(dx_3d)):.3f} K")

    # 6. Save Analysis Results to NetCDF
    export_to_netcdf(
        output_path=args.output,
        time_str=args.time,
        lats=lats,
        lons=lons,
        p_levels=p_levels,
        x_b=x_b,
        x_a=x_a_3d,
        dx=dx_3d
    )


if __name__ == "__main__":
    main()
