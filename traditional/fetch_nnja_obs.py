#!/usr/bin/env python3
"""
NNJA-AI Observation Downloader & NetCDF Exporter for GFS 3D-Var Pipeline
------------------------------------------------------------------------
Fetches ADPUPA conventional soundings and AMSU-A satellite radiances.
Reformats data for a Z-level DA system (state vector: t, td, u, v, p).
"""

import argparse
from datetime import datetime, timezone
import re
import numpy as np
import xarray as xr
from nnja_ai import DataCatalog

# Observation Error Standard Deviations (\sigma_o)
CONV_ERROR_DEFAULTS = {
    "t": 1.0,      # Temperature error (K)
    "td": 2.0,     # Dewpoint error (K)
    "u": 2.5,      # Zonal wind error (m/s)
    "v": 2.5,      # Meridional wind error (m/s)
    "p": 100.0,    # Pressure observation error (Pa)
}

# AMSU-A Channel Observation Errors (K) for Channels 1 to 15
AMSUA_CHANNEL_ERRORS = np.array([
    2.5, 2.2, 2.0, 0.6, 0.3, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.5, 2.5
], dtype=np.float32)


def fetch_conv_adpupa(catalog: DataCatalog, target_dt: datetime) -> xr.Dataset:
    date_str = target_dt.strftime("%Y-%m-%d")
    hour_val = target_dt.hour
    print(f"\n[1/2] Fetching conventional upper-air obs (NC002001) for {date_str} {hour_val:02d}:00 UTC...")

    try:
        ds_cat = catalog["conv-adpupa-NC002001"]
        df = ds_cat.sel(time=date_str).load_dataset(backend="pandas")

        if df is None or df.empty:
            print("  -> No matching conventional records found.")
            return None

        prlc_pattern = re.compile(r"^([A-Z0-9]+)_PRLC(\d+)$")
        col_grid = {}
        for col in df.columns:
            match = prlc_pattern.match(str(col))
            if match:
                raw_var, level_str = match.groups()
                level_val = float(level_str)  # Pressure in hPa
                if level_val not in col_grid:
                    col_grid[level_val] = {}
                col_grid[level_val][raw_var] = col

        if not col_grid:
            print("  -> Warning: No mandatory level columns (_PRLC) found.")
            return None

        lat_col = next((c for c in ["LAT", "latitude", "CLAT"] if c in df.columns), "LAT")
        lon_col = next((c for c in ["LON", "longitude", "CLON"] if c in df.columns), "LON")

        lats, lons, levels_hpa, vars_list, obs_vals, obs_errs = [], [], [], [], [], []

        def append_obs(v_type, vals, l_val_hpa, err_val):
            mask = ~np.isnan(vals)
            if np.any(mask):
                count = np.sum(mask)
                lats.extend(df[lat_col].values[mask])
                lons.extend(df[lon_col].values[mask])
                levels_hpa.extend(np.full(count, l_val_hpa, dtype=np.float32))
                vars_list.extend([v_type] * count)
                obs_vals.extend(vals[mask])
                obs_errs.extend([err_val] * count)

        for lvl_hpa, var_dict in col_grid.items():
            # 1. Temperature (t)
            t_vals = None
            if "TMDB" in var_dict:
                t_vals = df[var_dict["TMDB"]].values.astype(np.float32)
                append_obs("t", t_vals, lvl_hpa, CONV_ERROR_DEFAULTS["t"])

            # 2. Dewpoint (td = T - Dewpoint Depression)
            if "TMDP" in var_dict:
                dep_vals = df[var_dict["TMDP"]].values.astype(np.float32)
                td_vals = (t_vals - dep_vals) if t_vals is not None else dep_vals
                append_obs("td", td_vals, lvl_hpa, CONV_ERROR_DEFAULTS["td"])

            # 3. Pressure (p) - Sounding mandatory levels reported in hPa converted to Pa
            p_pa = np.full(len(df), lvl_hpa * 100.0, dtype=np.float32)
            append_obs("p", p_pa, lvl_hpa, CONV_ERROR_DEFAULTS["p"])

            # 4. Wind Components (u, v)
            if "UOB" in var_dict and "VOB" in var_dict:
                append_obs("u", df[var_dict["UOB"]].values.astype(np.float32), lvl_hpa, CONV_ERROR_DEFAULTS["u"])
                append_obs("v", df[var_dict["VOB"]].values.astype(np.float32), lvl_hpa, CONV_ERROR_DEFAULTS["v"])
            elif "WSPD" in var_dict and "WDIR" in var_dict:
                wspd = df[var_dict["WSPD"]].values.astype(np.float32)
                wdir = df[var_dict["WDIR"]].values.astype(np.float32)
                mask = ~np.isnan(wspd) & ~np.isnan(wdir) & (wspd >= 0) & (wdir >= 0)
                if np.any(mask):
                    rad = np.radians(wdir[mask])
                    u_calc = -wspd[mask] * np.sin(rad)
                    v_calc = -wspd[mask] * np.cos(rad)

                    count = len(u_calc)
                    lats.extend(df[lat_col].values[mask])
                    lons.extend(df[lon_col].values[mask])
                    levels_hpa.extend(np.full(count, lvl_hpa, dtype=np.float32))
                    vars_list.extend(["u"] * count)
                    obs_vals.extend(u_calc)
                    obs_errs.extend([CONV_ERROR_DEFAULTS["u"]] * count)

                    lats.extend(df[lat_col].values[mask])
                    lons.extend(df[lon_col].values[mask])
                    levels_hpa.extend(np.full(count, lvl_hpa, dtype=np.float32))
                    vars_list.extend(["v"] * count)
                    obs_vals.extend(v_calc)
                    obs_errs.extend([CONV_ERROR_DEFAULTS["v"]] * count)

        if not obs_vals:
            print("  -> Warning: No valid conventional observations extracted.")
            return None

        n_obs = len(obs_vals)
        print(f"  -> Extracted {n_obs} conventional observations with 'observation_error'.")

        ds_out = xr.Dataset(
            {
                "variable": ("obs", np.array(vars_list, dtype=str)),
                "observation_value": ("obs", np.array(obs_vals, dtype=np.float32)),
                "observation_error": ("obs", np.array(obs_errs, dtype=np.float32)),
                "latitude": ("obs", np.array(lats, dtype=np.float32)),
                "longitude": ("obs", np.array(lons, dtype=np.float32)),
                "level": ("obs", np.array(levels_hpa, dtype=np.float32)),  # Level in hPa
            }
        )

        return ds_out

    except Exception as e:
        print(f"  -> Error fetching conventional data: {e}")
        return None


def fetch_amsua(catalog: DataCatalog, target_dt: datetime) -> xr.Dataset:
    date_str = target_dt.strftime("%Y-%m-%d")
    print(f"\n[2/2] Fetching AMSU-A radiances (NC021023) for {date_str} around {target_dt.hour:02d}:00 UTC...")

    try:
        ds_cat = catalog["amsua-1bamua-NC021023"]
        df = ds_cat.sel(time=date_str).load_dataset(backend="pandas")

        if df is None or df.empty:
            print("  -> No matching AMSU-A radiance records found.")
            return None

        lat_col = next((c for c in df.columns if str(c).upper() in ["LAT", "LATITUDE", "CLAT"]), None)
        lon_col = next((c for c in df.columns if str(c).upper() in ["LON", "LONGITUDE", "CLON"]), None)
        sat_col = next((c for c in df.columns if str(c).upper() in ["SAID", "SAT_ID", "SATID"]), None)

        tb_cols = []
        for col in df.columns:
            s_col = str(col)
            if any(k in s_col.upper() for k in ["TMBR", "T2MB", "BRIT", "TB"]):
                if not any(sub in s_col.upper() for sub in ["QCPR", "QMAT", "CHN", "NUM"]):
                    tb_cols.append(col)

        tb_matrix = None
        if tb_cols:
            def get_chan_num(c):
                nums = re.findall(r"\d+", str(c))
                return int(nums[-1]) if nums else 0

            tb_cols = sorted(tb_cols, key=get_chan_num)
            if len(tb_cols) >= 15:
                tb_matrix = df[tb_cols].values.astype(np.float32)

        if tb_matrix is None:
            single_tb_col = next((c for c in df.columns if str(c).upper() in ["TMBR", "T2MB", "TB"]), None)
            if single_tb_col is not None:
                first_valid = df[single_tb_col].dropna()
                if not first_valid.empty and isinstance(first_valid.iloc[0], (list, np.ndarray)):
                    tb_matrix = np.vstack(df[single_tb_col].values).astype(np.float32)

        if tb_matrix is None:
            print("  -> Could not locate AMSU-A brightness temperature matrix.")
            return None

        n_obs, n_chan = tb_matrix.shape
        err_row = AMSUA_CHANNEL_ERRORS if n_chan == 15 else np.full(n_chan, 1.0, dtype=np.float32)
        err_matrix = np.tile(err_row, (n_obs, 1))

        data_vars = {
            "tb": (("obs_idx", "channel"), tb_matrix),
            "tb_error": (("obs_idx", "channel"), err_matrix),
            "observation_error": (("obs_idx", "channel"), err_matrix),
        }

        if sat_col is not None:
            data_vars["sat_id"] = ("obs_idx", df[sat_col].values.astype(str))

        lat_data = df[lat_col].values.astype(np.float32) if lat_col else np.zeros(n_obs, dtype=np.float32)
        lon_data = df[lon_col].values.astype(np.float32) if lon_col else np.zeros(n_obs, dtype=np.float32)

        print(f"  -> Extracted {n_obs} AMSU-A radiance profiles ({n_chan} channels) with observation errors.")

        return xr.Dataset(
            data_vars,
            coords={
                "latitude": ("obs_idx", lat_data),
                "longitude": ("obs_idx", lon_data),
                "channel": np.arange(1, n_chan + 1),
            },
        )

    except Exception as e:
        print(f"  -> Error fetching AMSU-A data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download NNJA-AI Observations with Errors for GFS 3D-Var")
    parser.add_argument("--datetime", type=str, default="2023-07-01T12:00:00")
    parser.add_argument("--out_conv", type=str, default="conv_adpupa_NC002001.nc")
    parser.add_argument("--out_amsua", type=str, default="amsua_NC021023.nc")

    args = parser.parse_args()
    target_dt = datetime.fromisoformat(args.datetime).replace(tzinfo=timezone.utc)

    catalog = DataCatalog()

    conv_ds = fetch_conv_adpupa(catalog, target_dt)
    if conv_ds is not None:
        conv_ds.to_netcdf(args.out_conv)
        print(f"  -> Saved conventional observations to '{args.out_conv}'")

    amsua_ds = fetch_amsua(catalog, target_dt)
    if amsua_ds is not None:
        amsua_ds.to_netcdf(args.out_amsua)
        print(f"  -> Saved AMSU-A radiance observations to '{args.out_amsua}'\n")


if __name__ == "__main__":
    main()
