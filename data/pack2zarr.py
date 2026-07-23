# convert_nc_to_zarr.py
import glob
import xarray as xr


def convert_icosahedral_nc_to_zarr(
    nc_pattern: str, zarr_out_path: str = "data/icosahedral_2023.zarr"
):
    print("Opening NetCDF files...")
    # Open dataset using xarray, combining across time
    ds = xr.open_mfdataset(
        nc_pattern, combine="nested", concat_dim="time", engine="netcdf4"
    )

    # Rechunk for optimal PyTorch DataLoader access
    # Chunking time=1 means loading 1 full global timestep at a time
    ds = ds.chunk(
        {"time": 1, "height": 32, "node": 2562, "face": 5120, "three": 3}
    )

    print(f"Writing unified Zarr archive to {zarr_out_path}...")
    ds.to_zarr(zarr_out_path, mode="w", consolidated=True)
    print("Zarr conversion complete.")


if __name__ == "__main__":
    convert_icosahedral_nc_to_zarr(
        "icosahedral_grid/global_icosahedral_*.nc"
    )
