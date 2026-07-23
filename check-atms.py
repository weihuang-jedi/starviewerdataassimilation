from nnja_ai import DataCatalog

catalog = DataCatalog()

# Access ATMS satellite radiance observations
atms_ds = catalog["atms-atms-NC021203"]
df_atms = atms_ds.sel(time="2021-01-01").load_dataset(backend="pandas")

# Select mandatory metadata and raw channel brightness temperatures
print("Available ATMS columns:", [col for col in df_atms.columns if "BRIT" in col or "LAT" in col or "LON" in col][:10])

# Satellite observations typically require:
# 1. Zenith angle / scan geometry filtering
# 2. Radiometric forward operator (e.g., RTTOV or PyRTTOV) to map model profiles to Brightness Temp (Tb)
