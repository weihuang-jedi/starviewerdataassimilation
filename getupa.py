import pandas as pd
import numpy as np
from nnja_ai import DataCatalog

# 1. Initialize the dataset catalog
catalog = DataCatalog()

# 2. Load upper-air radiosonde observations (ADPUPA) for a specific date
adpupa_ds = catalog["conv-adpupa-NC002001"]
df_obs = adpupa_ds.sel(time="2021-01-01").load_dataset(backend="pandas")

print(f"Loaded {len(df_obs)} sounding reports.")

# 3. Helper to extract a vertical profile array from nested UARLV structs
def extract_sounding_profile(row):
    uarlv_data = row["UARLV"]
    
    pressures, temps = [], []
    for level in uarlv_data:
        p = level.get("PRLC")
        # Extract temperature if available inside nested UATMP struct
        t = level.get("UATMP", {}).get("TMDB") if level.get("UATMP") else None
        
        # Filter sentinel missing value indicators (often 1e11 in BUFR)
        if p and p != 1e11 and t and t != 1e11:
            pressures.append(p / 100.0)  # Convert Pa to hPa
            temps.append(t)               # Kelvin
            
    return pd.DataFrame({"pressure_hpa": pressures, "obs_temp_k": temps})

# Example: Get sounding from the first station
single_sounding = extract_sounding_profile(df_obs.iloc[0])
print("\nFirst Station Sounding Profile:")
print(single_sounding.head())

