import numpy as np
import pandas as pd
from nnja_ai import DataCatalog

catalog = DataCatalog(mirror="gcp_brightband")
sfc_ds = catalog["conv-adpsfc-NC000001"]

# Load day subset
subset = sfc_ds.sel(time="2021-01-01")
df_sfc = subset.load_dataset(backend="pandas")

# 1. Inspect raw temperature values
print("--- Raw Observation Data Diagnostics ---")
print(f"Total rows loaded: {len(df_sfc)}")
print("TMPSQ1.TMDB summary stats:")
print(df_sfc["TMPSQ1.TMDB"].describe())

# 2. Convert OBS_DATE
if "OBS_DATE" in df_sfc.columns:
    df_sfc["OBS_DATE"] = pd.to_datetime(df_sfc["OBS_DATE"].astype(str), utc=True)
    start_time = pd.to_datetime("2021-01-01T11:00:00", utc=True)
    end_time = pd.to_datetime("2021-01-01T13:00:00", utc=True)
    df_sfc = df_sfc[(df_sfc["OBS_DATE"] >= start_time) & (df_sfc["OBS_DATE"] <= end_time)]
    print(f"Rows after 11Z-13Z time window filter: {len(df_sfc)}")

# 3. Clean missing sentinel values (< 1e5)
obs_df = df_sfc[df_sfc["TMPSQ1.TMDB"] < 1e5][["LAT", "LON", "TMPSQ1.TMDB"]].dropna().copy()

# Automatically adjust for Celsius if needed
sample_temp = obs_df["TMPSQ1.TMDB"].median()
if sample_temp < 100:  # Temp is in Celsius!
    print("Detected Celsius! Converting to Kelvin (+273.15)...")
    obs_df["TMPSQ1.TMDB"] = obs_df["TMPSQ1.TMDB"] + 273.15

obs_df.rename(columns={"TMPSQ1.TMDB": "y_obs"}, inplace=True)

# 4. Observation Operator
def mock_observation_operator(lats, lons):
    return 288.15 - 0.5 * np.abs(lats) + np.sin(np.radians(lons)) * 2.0

obs_df["Hx_b"] = mock_observation_operator(obs_df["LAT"].values, obs_df["LON"].values)
obs_df["innovation_d"] = obs_df["y_obs"] - obs_df["Hx_b"]

# 5. Quality Control
qc_threshold = 10.0
valid_obs = obs_df[np.abs(obs_df["innovation_d"]) <= qc_threshold]

print(f"\nFinal valid observations after QC: {len(valid_obs)}")
if not valid_obs.empty:
    print(valid_obs[["LAT", "LON", "y_obs", "Hx_b", "innovation_d"]].head())
