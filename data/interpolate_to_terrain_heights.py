#!/usr/bin/env python
import os
import argparse
import warnings
import numpy as np
import xarray as xr
import cfgrib

# Silence deprecation and runtime warnings
warnings.filterwarnings("ignore")
os.environ["GRIB_API_LOG_LEVEL"] = "0"


class TerrainHeightInterpolator:
    """
    Interpolates isobaric pressure-level fields to terrain-following geometric height levels:
        h(level, lat, lon) = H_max - eta(level) * (H_max - H_terrain(lat, lon))
    """

    def __init__(self, input_path: str, etopo_path: str, output_path: str):
        self.input_path = os.path.abspath(input_path)
        self.etopo_path = os.path.abspath(etopo_path)
        self.output_path = os.path.abspath(output_path)

        # Baseline flat-terrain target levels (meters)
        self.target_levels = np.array([
            2, 10, 20, 50, 75, 100, 150, 200, 300, 400,
            500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000,
            4500, 5000, 6000, 7000, 8000, 9000, 10000, 11500, 13000, 15000,
            17500, 20000
        ], dtype=np.float32)

        self.height_max = np.float32(self.target_levels[-1])  # 20000.0 m
        self.eta = (self.height_max - self.target_levels) / self.height_max  # eta coordinate [1.0 -> 0.0]

    def _load_and_align_topography(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Loads ETOPO dataset and interpolates/samples terrain elevation onto target grid."""
        print(f"[ETOPO] Extracting terrain topography from: {os.path.basename(self.etopo_path)}")

        ds_etopo = xr.open_dataset(self.etopo_path)
        z_var = ds_etopo['z']

        # Adjust longitude coordinates if ETOPO uses [-180, 180] and GFS uses [0, 360]
        lons_search = lons.copy()
        if (lons_search > 180).any() and (z_var['lon'].values < 0).any():
            lons_search = np.where(lons_search > 180, lons_search - 360, lons_search)

        # Nearest-neighbor interpolation of high-resolution ETOPO to GFS grid
        lat_dim = "lat" if "lat" in z_var.dims else "latitude"
        lon_dim = "lon" if "lon" in z_var.dims else "longitude"

        h_terrain = z_var.sel(
            {lat_dim: xr.DataArray(lats, dims="lat"),
             lon_dim: xr.DataArray(lons_search, dims="lon")},
            method="nearest"
        ).values

        ds_etopo.close()

        # Clip ocean/bathymetry depths to 0.0 meters for atmospheric terrain height
        h_terrain = np.clip(h_terrain, 0.0, None).astype(np.float32)
        return h_terrain

    def process_and_interpolate(self):
        print(f"[LOAD] Extracting source fields from GRIB: {os.path.basename(self.input_path)}")
        datasets = cfgrib.open_datasets(self.input_path, backend_kwargs={'errors': 'ignore'})

        # Gather and merge all datasets that contain 'isobaricInhPa' coordinates
        isobaric_parts = [ds for ds in datasets if 'isobaricInhPa' in ds.coords]
        if not isobaric_parts:
            raise KeyError("Could not locate any 'isobaricInhPa' pressure levels inside this file.")

        isobaric_ds = xr.merge(isobaric_parts, compat='override')
        available_vars = set(isobaric_ds.data_vars.keys())
        var_map = {}

        # Variable identification
        if 't' in available_vars: var_map['t'] = 't'
        elif 'tmp' in available_vars: var_map['t'] = 'tmp'

        if 'u' in available_vars: var_map['u'] = 'u'
        if 'v' in available_vars: var_map['v'] = 'v'

        if 'w' in available_vars: var_map['w'] = 'w'
        elif 'vvel' in available_vars: var_map['w'] = 'vvel'
        elif 'w_isobaric' in available_vars: var_map['w'] = 'w_isobaric'

        if 'q' in available_vars: var_map['q'] = 'q'
        elif 'shum' in available_vars: var_map['q'] = 'shum'
        elif 'q_isobaric' in available_vars: var_map['q'] = 'q_isobaric'
        elif 'r' in available_vars: var_map['r'] = 'r'

        if 'gh' in available_vars: var_map['gh'] = 'gh'
        else:
            raise KeyError(f"Geopotential Height ('gh') missing. Cannot calculate vertical coordinate profiles.")

        # Derive Specific Humidity (q) from Relative Humidity (r) if missing
        if 'q' not in var_map and 'r' in var_map:
            print(" -> 'q' missing on pressure levels. Deriving Specific Humidity from Relative Humidity ('r')...")
            t_k = isobaric_ds[var_map['t']].values
            r_pct = isobaric_ds[var_map['r']].values
            p_v = isobaric_ds['isobaricInhPa'].values
            p_hpa = p_v[:, np.newaxis, np.newaxis]

            t_c = t_k - 273.15
            e_s = 6.112 * np.exp((17.67 * t_c) / (t_k - 29.65))
            e = (r_pct / 100.0) * e_s
            q_values = np.clip((0.622 * e) / (p_hpa - (0.378 * e)), 1.0e-7, None)

            isobaric_ds['q'] = (isobaric_ds[var_map['t']].dims, q_values.astype(np.float32))
            isobaric_ds['q'].attrs = {
                "GRIB_shortName": "q",
                "units": "kg kg**-1",
                "long_name": "Specific Humidity Derived from Relative Humidity"
            }
            var_map['q'] = 'q'

        active_keys = [k for k in ['t', 'u', 'v', 'w', 'q'] if k in var_map]

        pressures = isobaric_ds['isobaricInhPa'].values.astype(np.float32)
        lats = isobaric_ds['latitude'].values
        lons = isobaric_ds['longitude'].values
        z_src = isobaric_ds[var_map['gh']].values.astype(np.float32)  # [levs, lats, lons]

        # Load terrain topography
        h_terrain = self._load_and_align_topography(lats, lons)  # [lats, lons]

        # Compute 3D target terrain-following heights: h = H_max - eta * (H_max - H_terrain)
        n_levs, n_lats, n_lons = z_src.shape
        n_targets = len(self.target_levels)

        eta_3d = self.eta[:, np.newaxis, np.newaxis]  # [n_targets, 1, 1]
        h_terrain_3d = h_terrain[np.newaxis, :, :]   # [1, n_lats, n_lons]
        target_h_3d = self.height_max - eta_3d * (self.height_max - h_terrain_3d)  # [n_targets, lats, lons]

        print("[INTERP] Executing fully vectorized vertical interpolation (fast)...")

        # Sort vertical profiles ascending by source height z_src
        sort_idx = np.argsort(z_src, axis=0)  # [n_levs, lats, lons]
        z_sorted = np.take_along_axis(z_src, sort_idx, axis=0)  # [n_levs, lats, lons]

        # Flatten spatial dimensions for vectorized batch interpolation: [n_levs, n_spatial]
        n_spatial = n_lats * n_lons
        z_2d = z_sorted.reshape(n_levs, n_spatial)
        target_h_2d = target_h_3d.reshape(n_targets, n_spatial)

        # Prepare pressure 2D array
        p_3d = np.broadcast_to(pressures[:, np.newaxis, np.newaxis], z_src.shape)
        p_sorted_2d = np.take_along_axis(p_3d, sort_idx, axis=0).reshape(n_levs, n_spatial)

        # Storage containers
        out_shape = (n_targets, n_lats, n_lons)
        interpolated_fields = {k: np.empty((n_targets, n_spatial), dtype=np.float32) for k in active_keys}
        interpolated_fields['p'] = np.empty((n_targets, n_spatial), dtype=np.float32)

        # --- FAST VECTORIZED 2D INTERPOLATION ---
        # Find upper bounding index for each target level in z_2d using searchsorted
        for t_idx in range(n_targets):
            h_target_layer = target_h_2d[t_idx, :]  # [n_spatial]

            # Vectorized binary search across sorted height profiles
            # Finds index i such that z_2d[i-1, s] <= h_target_layer[s] < z_2d[i, s]
            idx_upper = np.sum(z_2d < h_target_layer[np.newaxis, :], axis=0)
            idx_upper = np.clip(idx_upper, 1, n_levs - 1)
            idx_lower = idx_upper - 1

            s_indices = np.arange(n_spatial)

            z0 = z_2d[idx_lower, s_indices]
            z1 = z_2d[idx_upper, s_indices]
            dz = np.where(z1 == z0, 1.0e-5, z1 - z0)
            weights = (h_target_layer - z0) / dz  # [n_spatial]

            # Vectorized linear interpolation for pressure
            p0 = p_sorted_2d[idx_lower, s_indices]
            p1 = p_sorted_2d[idx_upper, s_indices]
            interpolated_fields['p'][t_idx, :] = p0 + weights * (p1 - p0)

            # Vectorized linear interpolation for active fields
            for key in active_keys:
                var_val = isobaric_ds[var_map[key]].values
                var_sorted_2d = np.take_along_axis(var_val, sort_idx, axis=0).reshape(n_levs, n_spatial)

                v0 = var_sorted_2d[idx_lower, s_indices]
                v1 = var_sorted_2d[idx_upper, s_indices]
                interp_val = v0 + weights * (v1 - v0)

                if key == 'q':
                    interp_val = np.clip(interp_val, 1.0e-7, None)

                interpolated_fields[key][t_idx, :] = interp_val

        # Reshape fields back to 3D grid [n_targets, n_lats, n_lons]
        for key in interpolated_fields:
            interpolated_fields[key] = interpolated_fields[key].reshape(out_shape)

        print("[PACKAGE] Structuring final NetCDF4 dataset container...")

        # Coordinates definition
        coords = {
            "level": ("level", np.arange(1, n_targets + 1, dtype=np.int32), {"long_name": "Model Level Index"}),
            "eta": ("level", self.eta, {"long_name": "Eta Coordinate Coefficient", "units": "1"}),
            "target_level": ("level", self.target_levels, {"long_name": "Baseline Flat-Terrain Height Level", "units": "meters"}),
            "latitude": ("latitude", lats, isobaric_ds['latitude'].attrs),
            "longitude": ("longitude", lons, isobaric_ds['longitude'].attrs),
        }

        if 'time' in isobaric_ds.coords:
            coords["time"] = isobaric_ds['time']

        # Data variables
        data_vars = {
            "h": (
                ["level", "latitude", "longitude"],
                target_h_3d,
                {"units": "meters", "long_name": "3D Terrain-Following Geometric Height Above Sea Level"}
            ),
            "h_terrain": (
                ["latitude", "longitude"],
                h_terrain,
                {"units": "meters", "long_name": "Surface Topography Elevation from ETOPO2022"}
            ),
            "p": (
                ["level", "latitude", "longitude"],
                interpolated_fields['p'],
                {"units": "hPa", "long_name": "Atmospheric Pressure at Target Level"}
            )
        }

        for target_key in active_keys:
            file_key = var_map[target_key]
            data_vars[target_key] = (
                ["level", "latitude", "longitude"],
                interpolated_fields[target_key],
                isobaric_ds[file_key].attrs
            )

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        ds_out = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "title": "GFS Variables Interpolated to Terrain-Following Height Coordinates",
                "terrain_formula": "h = Hmax - eta * (Hmax - Hterrain)",
                "height_max_meters": float(self.height_max)
            }
        )

        print(f"[SAVE] Saving output NetCDF file: {self.output_path}")
        ds_out.to_netcdf(self.output_path, format="NETCDF4")

        isobaric_ds.close()
        for ds in datasets:
            ds.close()
        ds_out.close()
        print("SUCCESS: Terrain-following vertical transformation complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Vertically re-map GFS datasets to terrain-following height levels using ETOPO topography."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input GFS GRIB2 file")
    parser.add_argument("-e", "--etopo", required=True, help="Path to ETOPO2022 NetCDF topography file")
    parser.add_argument("-o", "--output", required=True, help="Path to destination output NetCDF file")
    args = parser.parse_args()

    interpolator = TerrainHeightInterpolator(
        input_path=args.input,
        etopo_path=args.etopo,
        output_path=args.output
    )
    interpolator.process_and_interpolate()


if __name__ == "__main__":
    main()
