import numpy as np
import pandas as pd
from nnja_ai import DataCatalog

catalog = DataCatalog(mirror="gcp_brightband")
sfc_ds = catalog["conv-adpsfc-NC000001"]

# Load day subset
subset = sfc_ds.sel(time="2021-01-01")
df_sfc = subset.load_dataset(backend="pandas")

# 1. Parse OBS_DATE safely
df_sfc["OBS_DATE_DT"] = pd.to_datetime(df_sfc["OBS_DATE"].astype(str), utc=True, errors="coerce")

print("Min observation time in file:", df_sfc["OBS_DATE_DT"].min())
print("Max observation time in file:", df_sfc["OBS_DATE_DT"].max())

# 2. Filter for 11Z-13Z window
start_time = pd.to_datetime("2021-01-01T11:00:00Z")
end_time = pd.to_datetime("2021-01-01T13:00:00Z")

df_window = df_sfc[(df_sfc["OBS_DATE_DT"] >= start_time) & (df_sfc["OBS_DATE_DT"] <= end_time)].copy()

print(f"Rows matching 11Z-13Z window: {len(df_window)}")

# If 0 rows matched in that window, print the hourly breakdown to see when observations occur
if len(df_window) == 0:
    print("\nObservation count by hour for this date:")
    print(df_sfc["OBS_DATE_DT"].dt.hour.value_counts().sort_index())

# 3. Process DA Innovations if data is found
else:
    obs_df = df_window[df_window["TMPSQ1.TMDB"] < 1e5][["LAT", "LON", "TMPSQ1.TMDB"]].dropna().copy()
    obs_df.rename(columns={"TMPSQ1.TMDB": "y_obs"}, inplace=True)

    def mock_observation_operator(lats, lons):
        return 288.15 - 0.5 * np.abs(lats) + np.sin(np.radians(lons)) * 2.0

    obs_df["Hx_b"] = mock_observation_operator(obs_df["LAT"].values, obs_df["LON"].values)
    obs_df["innovation_d"] = obs_df["y_obs"] - obs_df["Hx_b"]

    qc_threshold = 10.0
    valid_obs = obs_df[np.abs(obs_df["innovation_d"]) <= qc_threshold]

    print(f"\nFinal valid observations after QC: {len(valid_obs)}")
    print(valid_obs[["LAT", "LON", "y_obs", "Hx_b", "innovation_d"]].head())
