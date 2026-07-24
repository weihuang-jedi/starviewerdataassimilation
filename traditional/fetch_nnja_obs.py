#!/usr/bin/env python3
"""
NNJA-AI Observation Downloader & NetCDF Exporter
------------------------------------------------
Queries NNJA-AI Cloud Parquet store on Google Cloud Storage for a specific
timestamp, extracts:
  1. Conventional upper-air sounding data (subtype NC002001)
  2. AMSU-A Level-1B satellite brightness temperatures (subtype NC021023)

Saves output as NetCDF files formatted for the 3D-Var pipeline.

Usage:
  python fetch_nnja_obs.py --datetime 2023-01-15T06:00:00
"""

import argparse
from datetime import datetime
import duckdb
import numpy as np
import xarray as xr

# NNJA-AI Public Bucket Path
BUCKET_BASE = "gs://gcp-nnja-ai/v1"


def setup_duckdb():
    """Configures DuckDB with spatial/HTTP extensions for direct GCS reads."""
    con = duckdb.connect(database=":memory:")

    # Try loading directly first; if not installed, run INSTALL
    try:
        con.execute("LOAD httpfs;")
    except duckdb.Error:
        con.execute("INSTALL httpfs; LOAD httpfs;")

    con.execute("SET gcs_access_key_id='';")  # Anonymous access to public bucket
    return con

#def setup_duckdb():
#    """Configures DuckDB with spatial/HTTP extensions for direct GCS reads."""
#    con = duckdb.connect(database=":memory:")
#    con.execute("INSTALL httpfs; LOAD httpfs;")
#    con.execute("SET gcs_access_key_id='';")  # Anonymous access to public bucket
#    return con


def fetch_conv_adpupa(con: duckdb.DuckDBPyConnection, target_dt: datetime, window_hours: int = 3):
    """Fetches conventional upper-air profile observations (NC002001)."""
    date_str = target_dt.strftime("%Y-%m-%d")
    hour_val = target_dt.hour

    print(f"[1/2] Fetching conventional upper-air obs (NC002001) for {date_str} {hour_val:02d}:00 UTC...")

    # Hive-partitioned GCS path pattern
    parquet_path = f"{BUCKET_BASE}/obs_type=conv/subtype=NC002001/date={date_str}/*.parquet"

    query = f"""
        SELECT 
            latitude, 
            longitude, 
            pressure as level, 
            observation_type as variable,
            observation_value,
            observation_error
        FROM read_parquet('{parquet_path}')
        WHERE abs(epoch_ms - {int(target_dt.timestamp() * 1000)}) <= {window_hours * 3600 * 1000}
    """

    try:
        df = con.execute(query).df()
        if df.empty:
            print("  -> No matching conventional records found in time window. Generating fallback format.")
            return None

        # Convert to xarray Dataset
        ds = xr.Dataset(
            {
                "observation_value": ("obs_idx", df["observation_value"].values.astype(np.float32)),
                "observation_error": ("obs_idx", df["observation_error"].values.astype(np.float32)),
                "variable": ("obs_idx", df["variable"].values.astype(str)),
            },
            coords={
                "latitude": ("obs_idx", df["latitude"].values.astype(np.float32)),
                "longitude": ("obs_idx", df["longitude"].values.astype(np.float32)),
                "level": ("obs_idx", df["level"].values.astype(np.float32)),
            },
        )
        return ds

    except Exception as e:
        print(f"  -> Error accessing GCS path ({e}). Returning None.")
        return None


def fetch_amsua(con: duckdb.DuckDBPyConnection, target_dt: datetime, window_hours: int = 3):
    """Fetches AMSU-A Level-1B satellite brightness temperature observations (NC021023)."""
    date_str = target_dt.strftime("%Y-%m-%d")

    print(f"[2/2] Fetching AMSU-A radiances (NC021023) for {date_str} around {target_dt.hour:02d}:00 UTC...")

    parquet_path = f"{BUCKET_BASE}/obs_type=satellite/subtype=NC021023/date={date_str}/*.parquet"

    # AMSU-A contains multi-channel brightness temperatures (typically 15 channels)
    query = f"""
        SELECT 
            latitude, 
            longitude, 
            brightness_temperatures,
            sat_id
        FROM read_parquet('{parquet_path}')
        WHERE abs(epoch_ms - {int(target_dt.timestamp() * 1000)}) <= {window_hours * 3600 * 1000}
    """

    try:
        df = con.execute(query).df()
        if df.empty:
            print("  -> No matching AMSU-A radiance records found in time window.")
            return None

        # Reconstruct 2D matrix (n_obs, 15 channels)
        tb_matrix = np.vstack(df["brightness_temperatures"].values).astype(np.float32)

        ds = xr.Dataset(
            {
                "brightness_temperature": (("obs_idx", "channel"), tb_matrix),
                "sat_id": ("obs_idx", df["sat_id"].values.astype(str)),
            },
            coords={
                "latitude": ("obs_idx", df["latitude"].values.astype(np.float32)),
                "longitude": ("obs_idx", df["longitude"].values.astype(np.float32)),
                "channel": np.arange(1, tb_matrix.shape[1] + 1),
            },
        )
        return ds

    except Exception as e:
        print(f"  -> Error querying GCS path ({e}). Returning None.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download NNJA-AI Observations to NetCDF")
    parser.add_argument(
        "--datetime",
        type=str,
        default="2023-07-01T12:00:00",
        help="Target datetime in ISO format, e.g. 2023-01-15T06:00:00",
    )
    parser.add_argument("--window", type=int, default=3, help="Time assimilation window (+/- hours)")
    parser.add_argument("--out_conv", type=str, default="conv_adpupa_NC002001.nc", help="Output path for conventional obs")
    parser.add_argument("--out_amsua", type=str, default="amsua_NC021023.nc", help="Output path for AMSU-A obs")

    args = parser.parse_args()
    target_dt = datetime.fromisoformat(args.datetime)

    con = setup_duckdb()

    # 1. Fetch & Write Conventional Obs
    conv_ds = fetch_conv_adpupa(con, target_dt, window_hours=args.window)
    if conv_ds is not None:
        conv_ds.to_netcdf(args.out_conv)
        print(f"  -> Saved conventional obs to: '{args.out_conv}'")

    # 2. Fetch & Write AMSU-A Radiance Obs
    amsua_ds = fetch_amsua(con, target_dt, window_hours=args.window)
    if amsua_ds is not None:
        amsua_ds.to_netcdf(args.out_amsua)
        print(f"  -> Saved AMSU-A radiance obs to: '{args.out_amsua}'")


if __name__ == "__main__":
    main()
