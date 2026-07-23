import numpy as np
import pandas as pd
from scipy.linalg import inv
from nnja_ai import DataCatalog

# --------------------------------------------------
# 1. FETCH NNJA-AI RADIOSONDE SOUNDING OBSERVATIONS
# --------------------------------------------------
print("Fetching upper-air sounding data from NNJA-AI...")
catalog = DataCatalog(mirror="gcp_brightband")

# Upper-air radiosonde observations
upa_ds = catalog["conv-adpupa-NC002001"]
subset = upa_ds.sel(time="2021-01-01")
df_upa = subset.load_dataset(backend="pandas")

# Extract the vertical profile for the first valid station in the file
def extract_sounding_profile(df):
    for _, row in df.iterrows():
        uarlv = row.get("UARLV")
        if not isinstance(uarlv, (list, np.ndarray)) or len(uarlv) == 0:
            continue
            
        pressures, temps = [], []
        for lvl in uarlv:
            p = lvl.get("PRLC")
            uatmp = lvl.get("UATMP")
            t = uatmp.get("TMDB") if isinstance(uatmp, dict) else None
            
            # Filter sentinel/missing values
            if p and p < 1e5 and t and t < 1e5 and t > 150:
                pressures.append(p / 100.0)  # Convert Pa to hPa
                temps.append(t)               # Kelvin
                
        if len(pressures) >= 5:
            profile_df = pd.DataFrame({"pressure_hpa": pressures, "temp_obs": temps})
            return profile_df.sort_values("pressure_hpa", ascending=False).reset_index(drop=True)
            
    raise ValueError("No valid sounding profile found in this partition.")

obs_profile = extract_sounding_profile(df_upa)
print(f"\nLoaded {len(obs_profile)} observation levels from radiosonde sounding.")

# --------------------------------------------------
# 2. DEFINE MODEL GRID & PRIOR BACKGROUND STATE x_b
# --------------------------------------------------
# Target model pressure levels (hPa)
grid_p = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0, 250.0, 200.0])
n_grid = len(grid_p)

# Mock Prior Model State x_b (Standard Atmosphere baseline in Kelvin)
x_b = 288.15 - 0.0065 * (1000.0 - grid_p) * 8.0 

# --------------------------------------------------
# 3. OBSERVATION OPERATOR H & OBSERVATION VECTOR y
# --------------------------------------------------
# Interpolate observations onto model grid to build simple linear observation map H
y_obs = np.interp(grid_p[::-1], obs_profile["pressure_hpa"][::-1], obs_profile["temp_obs"][::-1])[::-1]

# Linear identity matrix for collocated vertical levels
H = np.eye(n_grid)

# Innovation vector d = y - H(x_b)
innovation = y_obs - x_b

# --------------------------------------------------
# 4. COVARIANCE MATRICES B & R
# --------------------------------------------------
# Background error variance sigma_b = 2.0 K
sigma_b = 2.0
# Observation error variance sigma_r = 1.0 K (radiosondes are precise)
sigma_r = 1.0

# Construct Background Error Covariance B using Gaussian vertical correlation
L_p = 200.0  # Vertical correlation scale (hPa)
B = np.zeros((n_grid, n_grid))
for i in range(n_grid):
    for j in range(n_grid):
        dp = grid_p[i] - grid_p[j]
        B[i, j] = (sigma_b ** 2) * np.exp(-0.5 * (dp / L_p) ** 2)

# Construct Observation Error Covariance R
R = (sigma_r ** 2) * np.eye(n_grid)

# --------------------------------------------------
# 5. DATA ASSIMILATION SOLVER (OPTIMAL INTERPOLATION)
# --------------------------------------------------
# Compute Kalman Gain: K = B H^T (H B H^T + R)^(-1)
H_B_HT_plus_R = H @ B @ H.T + R
K = B @ H.T @ inv(H_B_HT_plus_R)

# Compute Analysis State: x_a = x_b + K * (y - H x_b)
x_a = x_b + K @ innovation

# Analysis Error Covariance: P_a = (I - K H) B
I = np.eye(n_grid)
P_a = (I - K @ H) @ B

# --------------------------------------------------
# 6. PRINT RESULTS
# --------------------------------------------------
da_results = pd.DataFrame({
    "Pressure (hPa)": grid_p,
    "Background x_b (K)": np.round(x_b, 2),
    "Observed y (K)": np.round(y_obs, 2),
    "Innovation (y-x_b)": np.round(innovation, 2),
    "Analysis x_a (K)": np.round(x_a, 2),
    "Analysis Var Red (%)": np.round((1 - np.diag(P_a) / np.diag(B)) * 100, 1)
})

print("\n--- Data Assimilation Results ---")
print(da_results.to_string(index=False))
