#!/usr/bin/env python3
"""
pack_icosahedral_zarr.py
------------------------
Memory-optimized, incremental streaming packer to convert large M6 unstructured
icosahedral weather NetCDF files into a consolidated Zarr dataset without OOM errors.
"""

import argparse
import glob
import os
import shutil
import warnings
import xarray as xr
import zarr
from numcodecs import Blosc

# Silence library notifications
warnings.filterwarnings("ignore")


class MemorySafeIcosahedralPacker:
    """
    Sequentially streams multi-file icosahedral NetCDF datasets into Zarr to prevent RAM exhaustion.
    """
    def __init__(self, file_pattern: str, output_zarr_path: str, time_chunk_size: int = 1):
        self.file_pattern = file_pattern
        self.output_zarr_path = output_zarr_path
        self.time_chunk_size = time_chunk_size

    def execute_conversion(self, compression_level: int = 3) -> None:
        matched_files = sorted(glob.glob(self.file_pattern))
        if not matched_files:
            raise FileNotFoundError(f"[ERROR] No files matched the pattern: '{self.file_pattern}'")

        print(f"[STAGE 1] Found {len(matched_files)} matching NetCDF files.")
        print(f"[STAGE 2] Target Zarr Store: '{self.output_zarr_path}' (Time Chunk Size = {self.time_chunk_size})")

        # Clean destination directory if present
        if os.path.exists(self.output_zarr_path):
            print(f" -> Removing existing Zarr store at '{self.output_zarr_path}' for clean initialization...")
            shutil.rmtree(self.output_zarr_path)

        compressor = Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.BITSHUFFLE)

        # Batch files into small chunks to avoid loading too many files at once
        batch_size = max(1, self.time_chunk_size)
        total_files = len(matched_files)

        for i in range(0, total_files, batch_size):
            file_batch = matched_files[i:i + batch_size]
            print(f" -> Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} ({len(file_batch)} files)...", flush=True)

            # Open current batch lazily
            ds_batch = xr.open_mfdataset(
                file_batch,
                concat_dim="time",
                combine="nested",
                data_vars="minimal",
                coords="minimal",
                compat="override"
            )

            # Determine vertical dimension
            vert_dim = 'level' if 'level' in ds_batch.dims else ('height' if 'height' in ds_batch.dims else None)

            # Set chunking
            chunk_spec = {'time': len(file_batch)}
            if vert_dim and vert_dim in ds_batch.dims:
                chunk_spec[vert_dim] = -1
            if 'node' in ds_batch.dims:
                chunk_spec['node'] = -1
            if 'face' in ds_batch.dims:
                chunk_spec['face'] = -1

            ds_batch = ds_batch.chunk(chunk_spec)

            encoding = {var: {'compressor': compressor} for var in ds_batch.data_vars}

            if i == 0:
                # Write initial store metadata and first batch
                ds_batch.to_zarr(
                    self.output_zarr_path,
                    mode='w',
                    encoding=encoding,
                    consolidated=True
                )
            else:
                # Append subsequent batches along the time dimension
                ds_batch.to_zarr(
                    self.output_zarr_path,
                    append_dim='time',
                    consolidated=True
                )

            ds_batch.close()

        print(f"\n[SUCCESS] Successfully compiled {total_files} NetCDF files into '{self.output_zarr_path}' without OOM issues!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-Safe Incremental NetCDF to Compressed Zarr Packer for M6 Datasets."
    )
    parser.add_argument("-i", "--input", default="icosahedral-grid/icosahedral_logstate_m6.202*.nc",
                        help="Input shell glob matching NetCDF files")
    parser.add_argument("-o", "--output", default="icosahedral_logstate.zarr",
                        help="Output Zarr target path")
    parser.add_argument("-c", "--chunk_size", type=int, default=4,
                        help="Batch chunk size for incremental writing (default: 4)")
    parser.add_argument("-l", "--level", type=int, default=3,
                        help="Zstd effort configuration [1-9] (default: 3)")

    args = parser.parse_args()

    packer = MemorySafeIcosahedralPacker(
        file_pattern=args.input,
        output_zarr_path=args.output,
        time_chunk_size=args.chunk_size
    )
    packer.execute_conversion(compression_level=args.level)


if __name__ == "__main__":
    main()
