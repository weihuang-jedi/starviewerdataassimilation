#!/usr/bin/env python3
"""
3D Spatial-Vertical 3D-Var Data Assimilation Script
---------------------------------------------------
- Observations: NNJA-AI Radiosonde Upper-Air Data (conv-adpupa-NC002001)
- Method: Preconditioned 3D-Var in v-control space (dx = L_3D * v)
- Localization: Gaspari-Cohn horizontal and vertical log-pressure tapering
- Quality Control: Temperature unit correction + 3-sigma innovation filter
- Output: CF-compliant NetCDF dataset via xarray
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from nnja_ai import DataCatalog

# ==============================================================================
# 1. FETCH RADIOSONDE UPPER-AIR OBS FROM NNJA-AI
# ==============================================================================
print("Fetching NNJA-AI upper-air (radiosonde) observations...")
catalog = DataCatalog(mirror="gcp_brightband")

ds_upr = catalog["conv-adpupa-NC002001"]
subset = ds_upr.sel(time="2021-01-01")
df_upr = subset.load_dataset(backend="pandas")

# Normalize Longitudes to [-180, 180]
df_upr["LON"] = np.where(df_upr["LON"] > 180, df_upr["LON"] - 360, df_upr["LON"])

# Detect temperature columns in dataset
temp_cols = [c for c in df_upr.columns if "TMDB" in c or "TMP" in c]

# Extract Multi-Level Profiles
obs_records = []
for idx, row in df_upr.iterrows():
    lat, lon = row["LAT"], row["LON"]
    for col in temp_cols:
        val = row[col]
        # Keep physically plausible temperatures in Celsius or Kelvin
        if pd.notna(val) and -100.0 < val < 350.0:
            p_level = 850.0  # Default fallback
            for p in [1000, 925, 850, 700, 500, 300, 250, 200, 100]:
                if str(p) in col:
                    p_level = float(p)
                    break
            obs_records.append({"LAT": lat, "LON": lon, "p_hpa": p_level, "y_obs": val})

obs_df = pd.DataFrame(obs_records)

# Filter spatial domain (CONUS bounding box)
if not obs_df.empty:
    spatial_mask = (
        (obs_df["LAT"] >= 30.0) & (obs_df["LAT"] <= 45.0) &
        (obs_df["LON"] >= -100.0) & (obs_df["LON"] <= -80.0)
    )
    obs_filtered = obs_df[spatial_mask].copy()
    if not obs_filtered.empty:
        obs_df = obs_filtered

# Fallback: Generate synthetic observations if no real data retrieved
if obs_df.empty:
    print("Warning: No matching observations found in domain. Using synthetic profiles...")
    synth_records = []
    for la, lo in zip([35.0, 38.0, 41.0], [-90.0, -85.0, -88.0]):
        for p in [1000.0, 850.0, 700.0, 500.0, 300.0]:
            t_synth = 288.15 * (p / 1000.0)**0.1903 + np.random.normal(0, 1.0)
            synth_records.append({"LAT": la, "LON": lo, "p_hpa": p, "y_obs": t_synth})
    obs_df = pd.DataFrame(synth_records)

# ------------------------------------------------------------------------------
# STEP A: UNIT CONVERSION (CELSIUS -> KELVIN)
# ------------------------------------------------------------------------------
# Convert Celsius temperatures (< 150 K) to Kelvin
celsius_mask = obs_df["y_obs"] < 150.0
if celsius_mask.any():
    n_converted = celsius_mask.sum()
    obs_df.loc[celsius_mask, "y_obs"] += 273.15
    print(f"Unit Correction: Converted {n_converted} observations from Celsius to Kelvin.")

# Subsample dataset for fast execution
n_sample = min(40, len(obs_df))
obs_df = obs_df.sample(n=n_sample, random_state=42).reset_index(drop=True)
n_obs = len(obs_df)

print(f"Extracted {n_obs} multi-level observation points.")

# ==============================================================================
# 2. DEFINE 3D SPATIAL & VERTICAL GRID
# ==============================================================================
lats = np.linspace(32.0, 42.0, 5)
lons = np.linspace(-95.0, -85.0, 5)
p_levels = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])  # hPa

n_lat, n_lon, n_lev = len(lats), len(lons), len(p_levels)
grid_lon, grid_lat, grid_p = np.meshgrid(lons, lats, p_levels, indexing="ij")

grid_coords = np.column_stack([
    grid_lat.ravel(), 
    grid_lon.ravel(), 
    grid_p.ravel()
])
n_grid_2d = n_lat * n_lon
n_grid_3d = len(grid_coords)

coords_2d = np.column_stack([grid_lat[:, :, 0].ravel(), grid_lon[:, :, 0].ravel()])

# Standard atmosphere background state x_b
p_ref = 1000.0
x_b_1d = 288.15 * (p_levels / p_ref) ** 0.1903
x_b = np.tile(x_b_1d, n_grid_2d)

# ==============================================================================
# 3. GASPARI-COHN LOCALIZATION FUNCTION
# ==============================================================================
def gaspari_cohn(r):
    c = np.zeros_like(r)
    m1 = (r >= 0) & (r < 1)
    c[m1] = 1.0 - 5/3*r[m1]**2 + 5/8*r[m1]**3 + 0.5*r[m1]**4 - 0.25*r[m1]**5
    m2 = (r >= 1) & (r < 2)
    c[m2] = (
        4/3*r[m2]**(-1) - 5.0 + 5.0*r[m2] - 5/3*r[m2]**2 
        + 5/8*r[m2]**3 - 0.5*r[m2]**4 + 1/12*r[m2]**5
    )
    return np.clip(c, 0.0, 1.0)

# ==============================================================================
# 4. CONSTRUCT HORIZONTAL & VERTICAL B MATRICES
# ==============================================================================
# Horizontal Covariance (B_H) + Localization
dists_2d = cdist(coords_2d, coords_2d) * 111.0  # km
sigma_b_h = 2.0
L_h = 250.0      # Spatial correlation length (km)
L_loc_h = 200.0  # Horizontal localization radius (km)

B_H_raw = (sigma_b_h**2) * np.exp(-0.5 * (dists_2d / L_h)**2)
C_H = gaspari_cohn(dists_2d / L_loc_h)
B_H_loc = B_H_raw * C_H + 1e-4 * np.eye(n_grid_2d)
L_H = np.linalg.cholesky(B_H_loc)

# Vertical Covariance (B_V) + Log-Pressure Localization
ln_p = np.log(p_levels)
dists_v = np.abs(ln_p[:, None] - ln_p[None, :])
sigma_b_v = 1.0
L_v = 0.8       # Vertical correlation scale (log-pressure)
L_loc_v = 0.6   # Vertical localization radius (log-pressure)

B_V_raw = (sigma_b_v**2) * np.exp(-0.5 * (dists_v / L_v)**2)
C_V = gaspari_cohn(dists_v / L_loc_v)
B_V_loc = B_V_raw * C_V + 1e-4 * np.eye(n_lev)
L_V = np.linalg.cholesky(B_V_loc)

# 3D Cholesky Factor via Kronecker Product: L_3D = L_H (x) L_V
L_3D = np.kron(L_H, L_V)

# ==============================================================================
# 5. FORWARD OPERATOR H & 3-SIGMA QC FILTER
# ==============================================================================
obs_coords = obs_df[["LAT", "LON", "p_hpa"]].values
obs_latlon = obs_coords[:, :2]
obs_p = obs_coords[:, 2]

grid_latlon = grid_coords[:, :2]
grid_p_vals = grid_coords[:, 2]

h_dists = cdist(obs_latlon, grid_latlon) * 111.0
v_dists = np.abs(np.log(obs_p[:, None]) - np.log(grid_p_vals[None, :]))

# Distance-weighted 3D forward operator
weights = np.exp(-0.5 * (h_dists / 60.0)**2 - 0.5 * (v_dists / 0.3)**2)
H = weights / weights.sum(axis=1, keepdims=True)

y = obs_df["y_obs"].values
sigma_r = 1.5  # Observation error standard deviation (K)

# Compute raw innovations: d_raw = y - H(x_b)
d_raw = y - H @ x_b

# ------------------------------------------------------------------------------
# STEP B: 3-SIGMA INNOVATION QUALITY CONTROL (QC)
# ------------------------------------------------------------------------------
sigma_b_total = np.sqrt(sigma_b_h**2 + sigma_b_v**2)
sigma_total = np.sqrt(sigma_b_total**2 + sigma_r**2)
qc_threshold = 3.0 * sigma_total  # ~8.08 K threshold

qc_mask = np.abs(d_raw) <= qc_threshold
n_rejected = np.sum(~qc_mask)

print(f"QC Filter (3-Sigma = {qc_threshold:.2f} K): Kept {np.sum(qc_mask)}/{n_obs} obs (Rejected {n_rejected}).")

# Apply QC mask to filter observations, forward operator, and innovations
y = y[qc_mask]
H = H[qc_mask, :]
d = d_raw[qc_mask]
n_obs_passed = len(y)

R_inv = (1.0 / sigma_r**2) * np.eye(n_obs_passed)

# ==============================================================================
# 6. INCREMENTAL 3D-VAR IN v-SPACE
# ==============================================================================
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

# Convert v back to physical state increment (dx = L_3D * v)
v_opt = res.x
dx_3d = L_3D @ v_opt
x_a_3d = x_b + dx_3d

# ==============================================================================
# 7. SUMMARY PRINT
# ==============================================================================
grid_df = pd.DataFrame({
    "Lat": grid_coords[:, 0],
    "Lon": grid_coords[:, 1],
    "p_hPa": grid_coords[:, 2],
    "x_b (K)": np.round(x_b, 2),
    "x_a (K)": np.round(x_a_3d, 2),
    "dx (K)": np.round(dx_3d, 2)
})

print(f"\n3D-Var Converged: {res.success} in {res.nit} iterations.")
print(f"Final Cost J: {res.fun:.3f}")
print(f"Overall Mean Abs Increment: {np.mean(np.abs(dx_3d)):.3f} K")

print("\n--- Average Increments across Pressure Levels ---")
vertical_summary = grid_df.groupby("p_hPa")[["x_b (K)", "dx (K)"]].mean().reset_index()
print(vertical_summary.to_string(index=False))

print("\nSample 3D Grid Points at 850 hPa:")
print(grid_df[grid_df["p_hPa"] == 850.0].head(8).to_string(index=False))

# ==============================================================================
# 8. EXPORT 3D ANALYSIS TO STRUCTURED NETCDF
# ==============================================================================
print("\nExporting 3D Analysis state to NetCDF...")

# Reshape vectors to 3D arrays: (n_lat, n_lon, n_lev)
x_b_3d_arr = x_b.reshape(n_lat, n_lon, n_lev)
x_a_3d_arr = x_a_3d.reshape(n_lat, n_lon, n_lev)
dx_3d_arr = dx_3d.reshape(n_lat, n_lon, n_lev)

analysis_time = pd.Timestamp("2021-01-01T00:00:00")

ds_out = xr.Dataset(
    data_vars={
        "x_b": (["time", "latitude", "longitude", "level"], x_b_3d_arr[None, ...], 
                {"standard_name": "air_temperature", "long_name": "Background State (x_b)", "units": "K"}),
        "x_a": (["time", "latitude", "longitude", "level"], x_a_3d_arr[None, ...], 
                {"standard_name": "air_temperature", "long_name": "3D-Var Analysis State (x_a)", "units": "K"}),
        "dx":  (["time", "latitude", "longitude", "level"], dx_3d_arr[None, ...], 
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

nc_filename = "3dvar_analysis_20210101.nc"
ds_out.to_netcdf(nc_filename)
print(f"Saved NetCDF file '{nc_filename}' successfully.")
