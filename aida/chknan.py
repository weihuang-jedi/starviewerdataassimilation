import xarray as xr

ds = xr.open_dataset('cycling_output_20230306/aida_analysis_20230306_t06z.nc')

print("Latitude preview:", ds['latitude'].values[:5])
print("Longitude preview:", ds['longitude'].values[:5])
print("Any NaNs in lat?", float(ds['latitude'].isnull().sum()))
print("Any NaNs in lon?", float(ds['longitude'].isnull().sum()))
