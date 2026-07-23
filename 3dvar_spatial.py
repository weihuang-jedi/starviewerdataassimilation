import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from nnja_ai import DataCatalog

import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. FETCH MULTI-STATION SURFACE OBS FROM NNJA-AI
# --------------------------------------------------
print("Fetching NNJA-AI surface observations...")
catalog = DataCatalog(mirror="gcp_brightband")

sfc_ds = catalog["conv-adpsfc-NC000001"]
subset = sfc_ds.sel(time="2021-01-01")
df_sfc = subset.load_dataset(backend="pandas")

# Filter valid temperature reports
obs_df = df_sfc[df_sfc["TMPSQ1.TMDB"] < 1e5][["LAT", "LON", "TMPSQ1.TMDB"]].dropna().copy()
obs_df["LON"] = np.where(obs_df["LON"] > 180, obs_df["LON"] - 360, obs_df["LON"])

spatial_mask = (
    (obs_df["LAT"] >= 30.0) & (obs_df["LAT"] <= 45.0) &
    (obs_df["LON"] >= -100.0) & (obs_df["LON"] <= -80.0)
)
obs_filtered = obs_df[spatial_mask].copy()

if len(obs_filtered) == 0:
    obs_filtered = obs_df.copy()

n_sample = min(20, len(obs_filtered))
obs_df = obs_filtered.sample(n=n_sample, random_state=42).reset_index(drop=True)
obs_df.rename(columns={"TMPSQ1.TMDB": "y_obs"}, inplace=True)
n_obs = len(obs_df)

print(f"Selected {n_obs} observations.")

# --------------------------------------------------
# 2. SPATIAL GRID & PRIOR BACKGROUND x_b
# --------------------------------------------------
lat_min, lat_max = obs_df["LAT"].min() - 1, obs_df["LAT"].max() + 1
lon_min, lon_max = obs_df["LON"].min() - 1, obs_df["LON"].max() + 1

lats = np.linspace(lat_min, lat_max, 8)
lons = np.linspace(lon_min, lon_max, 8)
grid_lon, grid_lat = np.meshgrid(lons, lats)
grid_coords = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
n_grid = len(grid_coords)

# Background State x_b
x_b = 288.15 - 0.4 * (grid_coords[:, 0] - lat_min)

# --------------------------------------------------
# 3. GASPARI-COHN LOCALIZATION & B matrix
# --------------------------------------------------
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

grid_dists = cdist(grid_coords, grid_coords) * 111.0

sigma_b = 2.0  # Background error std dev (K)
L_b = 200.0    # Gaussian length scale (km)
B_raw = (sigma_b ** 2) * np.exp(-0.5 * (grid_dists / L_b) ** 2)

L_loc = 150.0  # Localization half-radius (km)
C_loc = gaspari_cohn(grid_dists / L_loc)
B_loc = B_raw * C_loc

# Add small regularization to ensure positive-definiteness for Cholesky
B_loc += 1e-4 * np.eye(n_grid)

# Compute B^{1/2} using Cholesky Decomposition: B = L @ L.T
L_B = np.linalg.cholesky(B_loc)

# --------------------------------------------------
# 4. FORWARD OPERATOR H & INNOVATION d
# --------------------------------------------------
obs_coords = obs_df[["LAT", "LON"]].values
obs_grid_dists = cdist(obs_coords, grid_coords) * 111.0

# Gaussian distance weights for H
weights = np.exp(-0.5 * (obs_grid_dists / 50.0)**2)
H = weights / weights.sum(axis=1, keepdims=True)

y = obs_df["y_obs"].values
sigma_r = 1.0  # Observation error std dev (K)
R_inv = (1.0 / sigma_r**2) * np.eye(n_obs)

# Innovation vector: d = y - H(x_b)
d = y - H @ x_b

# --------------------------------------------------
# 5. INCREMENTAL 3D-VAR IN v-SPACE
# --------------------------------------------------
# Transformation operator: G = H @ L_B
G = H @ L_B

def cost_function_v(v):
    # J(v) = 0.5 * v^T v + 0.5 * (d - G v)^T R^{-1} (d - G v)
    residual = d - G @ v
    j_b = 0.5 * np.dot(v, v)
    j_o = 0.5 * residual.T @ R_inv @ residual
    return j_b + j_o

def cost_gradient_v(v):
    residual = d - G @ v
    return v - G.T @ R_inv @ residual

# Initial guess in control space
v0 = np.zeros(n_grid)

print("Solving Incremental 3D-Var in v-space...")
res = minimize(
    fun=cost_function_v,
    x0=v0,
    jac=cost_gradient_v,
    method="L-BFGS-B",
    options={"gtol": 1e-5, "maxiter": 100}
)

# Convert control variable v back to physical state increment dx
v_opt = res.x
dx = L_B @ v_opt
x_a = x_b + dx

# --------------------------------------------------
# 6. RESULTS
# --------------------------------------------------
results_df = pd.DataFrame({
    "Grid Lat": np.round(grid_coords[:, 0], 2),
    "Grid Lon": np.round(grid_coords[:, 1], 2),
    "Background x_b (K)": np.round(x_b, 2),
    "Analysis x_a (K)": np.round(x_a, 2),
    "Increment dx (K)": np.round(dx, 2)
})

print(f"\n3D-Var Converged: {res.success} in {res.nit} iterations.")
print(f"Final Cost J: {res.fun:.3f}")
print(f"Mean Abs Increment: {np.mean(np.abs(dx)):.3f} K")
print(f"Max Increment: {np.max(dx):.3f} K")
print(f"Min Increment: {np.min(dx):.3f} K")

print("\nSample Grid Points After 3D-Var Assimilation:")
print(results_df.head(10).to_string(index=False))

# --------------------------------------------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(grid_coords[:, 1], grid_coords[:, 0], c=dx, cmap="coolwarm", s=100, edgecolors="k")
plt.colorbar(sc, label="Analysis Increment dx (K)")
plt.scatter(obs_df["LON"], obs_df["LAT"], color="black", marker="x", s=50, label="Observations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("3D-Var Surface Temperature Analysis Increments (Gaspari-Cohn)")
plt.legend()
plt.show()
