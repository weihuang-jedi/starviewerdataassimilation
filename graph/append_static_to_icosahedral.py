import argparse
import os
import xarray as xr
import numpy as np

class IcosahedralStaticAppender:
    """
    Utility module to sample, interpolate, and append fixed geographic features
    (elevation height and derived land-sea masks) directly onto the 1D nodes of 
    an unstructured global icosahedral mesh file.
    """
    def __init__(self, mesh_path: str, etopo_path: str):
        self.mesh_path = mesh_path
        self.etopo_path = etopo_path
        self.ds_mesh = None
        self.ds_etopo = None

    def execute_mapping(self) -> None:
        print(f"[STAGE 1] Loading unstructured icosahedral mesh: {self.mesh_path}")
        self.ds_mesh = xr.open_dataset(self.mesh_path).load()

        print(f"[STAGE 2] Opening ETOPO digital elevation model: {self.etopo_path}")
        self.ds_etopo = xr.open_dataset(self.etopo_path)

        # 1. Isolate icosahedral coordinate vertices
        mesh_lats = self.ds_mesh['latitude'].values
        mesh_lons = self.ds_mesh['longitude'].values
        
        # Enforce 0-360 longitude alignment matching ETOPO standard grid bounds
        mesh_lon_converted = np.mod(mesh_lons, 360)

        # 2. Package destination coordinates into xarray lookup structures
        target_lon = xr.DataArray(mesh_lon_converted, dims=["node"])
        target_lat = xr.DataArray(mesh_lats, dims=["node"])

        # 3. Detect coordinate keys inside the ETOPO dataset dynamically
        etopo_lon_key = 'lon' if 'lon' in self.ds_etopo.coords else 'longitude'
        etopo_lat_key = 'lat' if 'lat' in self.ds_etopo.coords else 'latitude'
        elv_var = 'z' if 'z' in self.ds_etopo.data_vars else 'elevation'

        print(f"[STAGE 3] Interpolating topographic altitude mapping via variable '{elv_var}'...")
        # Linear interpolation extracts smooth, area-weighted heights at the precise icosahedral vertex locations
        node_elevation = self.ds_etopo[elv_var].interp(
            {etopo_lon_key: target_lon, etopo_lat_key: target_lat}, 
            method="linear"
        ).values
        node_elevation = np.nan_to_num(node_elevation, nan=0.0)

        # 4. Derive binary land-sea mask natively from structural relief bounds
        print("[STAGE 4] Deriving physical land-sea binary masks...")
        node_land_mask = np.where(node_elevation >= 0.0, 1.0, 0.0)

        # 5. Inject variables cleanly back into the master dataset structure
        print("[STAGE 5] Packaging updated attributes into UGRID NetCDF...")
        self.ds_mesh['land_sea_mask'] = (
            ["node"], 
            node_land_mask.astype(np.float32), 
            {
                "long_name": "Land-Sea Binary Mask Derived from ETOPO (1=Land, 0=Ocean)", 
                "units": "fraction",
                "coordinates": "longitude latitude",
                "mesh": "icosahedral_mesh"
            }
        )
        self.ds_mesh['elevation'] = (
            ["node"], 
            node_elevation.astype(np.float32), 
            {
                "long_name": "Topographic Elevation Height Above Sea Level", 
                "units": "meters",
                "coordinates": "longitude latitude",
                "mesh": "icosahedral_mesh"
            }
        )

        self.ds_etopo.close()

        # Write safely using an atomic staging file mirror to prevent data corruption
        staging_path = self.mesh_path + ".tmp_staging"
        print(f"[SAVE] Saving updated geometry file to staging area: {staging_path}")
        self.ds_mesh.to_netcdf(staging_path, format="NETCDF4")
        self.ds_mesh.close()

        os.replace(staging_path, self.mesh_path)
        print(f"SUCCESS: Icosahedral mesh file '{self.mesh_path}' now permanently contains land_sea_mask and elevation properties!\n")


def main():
    parser = argparse.ArgumentParser(description="Map static ETOPO landscape values onto a spherical icosahedral mesh.")
    parser.add_argument("-m", "--mesh", required=True, help="Path to your global_icosahedral_mesh_m*.nc file")
    parser.add_argument("-e", "--etopo", required=True, help="Path to your raw NOAA ETOPO NetCDF file")
    args = parser.parse_args()

    appender = IcosahedralStaticAppender(mesh_path=args.mesh, etopo_path=args.etopo)
    appender.execute_mapping()

if __name__ == "__main__":
    main()
