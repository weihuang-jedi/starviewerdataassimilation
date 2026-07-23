import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from nnja_ai import DataCatalog

# --------------------------------------------------
# 1. FETCH MULTI-STATION SURFACE OBS FROM NNJA-AI
# --------------------------------------------------
print("Fetching NNJA-AI surface observations...")
catalog = DataCatalog(mirror="gcp_brightband")

sfc_ds = catalog["conv-adpsfc-NC000001"]
subset = sfc_ds.sel(time="2021-01-01")
df_sfc = subset.load_dataset(backend="pandas")

# Clean observation dataset and filter valid temperatures (TMPSQ1.TMDB)
obs_df = df_sfc[df_sfc["TMPSQ1.TMDB"] < 1e5][["LAT", "LON", "TMPSQ1.TMDB"]].dropna().copy()

# Convert Longitudes to [-180, 180] if stored in [0, 360]
obs_df["LON"] = np.where(obs_df["LON"] > 180, obs_df["LON"] - 360, obs_df["LON"])

# Spatial bounding box (CONUS: Lat 30 to 45, Lon -100 to -80)
spatial_mask = (
    (obs_df["LAT"] >= 30.0) & (obs_df["LAT"] <= 45.0) &
    (obs_df["LON"] >= -100.0) & (obs_df["LON"] <= -80.0)
)
obs_filtered = obs_df[spatial_mask].copy()

# Fallback: If no stations in that box, take a sample from any available coordinates
if len(obs_filtered) == 0:
    print("No observations matched exact lat/lon box. Falling back to global sample...")
    obs_filtered = obs_df.copy()

# Safely take up to 50 observations
n_sample = min(50, len(obs_filtered))
obs_df = obs_filtered.sample(n=n_sample, random_state=42).reset_index(drop=True)
obs_df.rename(columns={"TMPSQ1.TMDB": "y_obs"}, inplace=True)

n_obs = len(obs_df)
print(f"Successfully selected {n_obs} observations.")

# --------------------------------------------------
# 2. DEFINE SPATIAL GRID & PRIOR BACKGROUND STATE x_b
# --------------------------------------------------
lat_min, lat_max = obs_df["LAT"].min() - 1, obs_df["LAT"].max() + 1
lon_min, lon_max = obs_df["LON"].min() - 1, obs_df["LON"].max() + 1

lats = np.linspace(lat_min, lat_max, 10)
lons = np.linspace(lon_min, lon_max, 10)
grid_lon, grid_lat = np.meshgrid(lons, lats)

grid_coords = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
n_grid = len(grid_coords)

# Background temperature profile x_b (288 K with latitude gradient)
x_b = 288.15 - 0.4 * (grid_coords[:, 0] - lat_min)

# --------------------------------------------------
# 3. GASPARI-COHN LOCALIZATION & B MATRIX
# --------------------------------------------------
def gaspari_cohn(r):
    """Computes Gaspari-Cohn tapering scalar for normalized distance r = d / L_loc."""
    c = np.zeros_like(r)
    m1 = (r >= 0) & (r < 1)
    c[m1] = 1.0 - 5/3*r[m1]**2 + 5/8*r[m1]**3 + 0.5*r[m1]**4 - 0.25*r[m1]**5
    
    m2 = (r >= 1) & (r < 2)
    c[m2] = (
        4/3*r[m2]**(-1) - 5.0 + 5.0*r[m2] - 5/3*r[m2]**2 
        + 5/8*r[m2]**3 - 0.5*r[m2]**4 + 1/12*r[m2]**5
    )
    return np.clip(c, 0.0, 1.0)

# Distance matrix (approx. km)
grid_dists = cdist(grid_coords, grid_coords) * 111.0

sigma_b = 2.5   # Kelvin
L_b = 300.0     # Covariance length scale (km)
B_raw = (sigma_b ** 2) * np.exp(-0.5 * (grid_dists / L_b) ** 2)

L_loc = 250.0   # Localization half-radius (km)
C_loc = gaspari_cohn(grid_dists / L_loc)
B_loc = B_raw * C_loc  # Hadamard product

B_inv = np.linalg.inv(B_loc + 1e-6 * np.eye(n_grid))

# --------------------------------------------------
# 4. FORWARD OBSERVATION OPERATOR H
# --------------------------------------------------
obs_coords = obs_df[["LAT", "LON"]].values
obs_grid_dists = cdist(obs_coords, grid_coords) * 111.0

# Inverse-distance weighting operator H (maps n_grid -> n_obs)
weights = 1.0 / (obs_grid_dists + 1.0)**2
H = weights / weights.sum(axis=1, keepdims=True)

y = obs_df["y_obs"].values
sigma_r = 1.0  # Sensor error (Kelvin)
R_inv = (1.0 / sigma_r**2) * np.eye(n_obs)

# --------------------------------------------------
# 5. 3D-VAR OPTIMIZATION
# --------------------------------------------------
def cost_function(x):
    dx = x - x_b
    dy = y - H @ x
    return 0.5 * dx.T @ B_inv @ dx + 0.5 * dy.T @ R_inv @ dy

def cost_gradient(x):
    dx = x - x_b
    dy = y - H @ x
    return B_inv @ dx - H.T @ R_inv @ dy

print("Solving 3D-Var optimization...")
res = minimize(
    fun=cost_function,
    x0=x_b,
    jac=cost_gradient,
    method="L-BFGS-B",
    options={"gtol": 1e-5, "maxiter": 200}
)

x_a = res.x
increments = x_a - x_b

# --------------------------------------------------
# 6. RESULTS
# --------------------------------------------------
results_df = pd.DataFrame({
    "Grid Lat": np.round(grid_coords[:, 0], 2),
    "Grid Lon": np.round(grid_coords[:, 1], 2),
    "Background x_b (K)": np.round(x_b, 2),
    "Analysis x_a (K)": np.round(x_a, 2),
    "Increment (x_a - x_b)": np.round(increments, 2)
})

print(f"\n3D-Var Converged: {res.success} in {res.nit} iterations.")
print(f"Mean Abs Increment: {np.mean(np.abs(increments)):.3f} K")
print(f"Max Positive Increment: {np.max(increments):.3f} K")
print(f"Max Negative Increment: {np.min(increments):.3f} K")

print("\nSample Grid Points After 3D-Var Assimilation:")
print(results_df.head(10).to_string(index=False))
