#!/usr/bin/env python
import argparse
import os
import xarray as xr
import zarr
import warnings
from numcodecs import Blosc

# Silence library notifications
warnings.filterwarnings("ignore")

class IcosahedralToZarrPacker:
    """
    An object-oriented packaging engine designed to compile large multi-file collections
    of multi-level (height) unstructured icosahedral weather NetCDF files into an optimized,
    cloud-native, and ML-ready consolidated Zarr dataset.
    """
    def __init__(self, file_pattern: str, output_zarr_path: str, time_chunk_size: int = 32):
        """
        Args:
            file_pattern (str): Shell glob pattern matching source NetCDF files
            output_zarr_path (str): Target directory destination path for the compiled Zarr store
            time_chunk_size (int): Temporal sequence step length inside each chunk (default: 32)
        """
        self.file_pattern = file_pattern
        self.output_zarr_path = output_zarr_path
        self.time_chunk_size = time_chunk_size
        self.ds = None

    def execute_conversion(self, compression_level: int = 3) -> None:
        """
        Loads files lazily, builds the targeted multi-dimensional chunk graph, binds
        high-performance Blosc compression maps, and writes the Zarr store.
        """
        print(f"[STAGE 1] Resolving multi-file netCDF collection pattern: {self.file_pattern}")

        # Open files along the time dimension lazily
        self.ds = xr.open_mfdataset(
            self.file_pattern,
            concat_dim="time",
            combine="nested",
            data_vars="minimal",
            coords="minimal",
            compat="override"
        )

        print("[STAGE 2] Enforcing training chunk boundaries (time, height, node alignment)...")
        
        # Enforce time chunking on meteorological fields while keeping spatial and vertical dimensions unified
        chunk_spec = {'time': self.time_chunk_size}
        if 'height' in self.ds.dims:
            chunk_spec['height'] = -1
        if 'node' in self.ds.dims:
            chunk_spec['node'] = -1
            
        self.ds = self.ds.chunk(chunk_spec)

        print(f"[STAGE 3] Building Blosc ZStandard encoding profiles (effort level={compression_level})...")
        # Configure high-efficiency BitShuffle compression mapping used by modern deep learning architectures
        compressor = Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
        encoding = {var: {'compressor': compressor} for var in self.ds.data_vars}

        print(f"[STAGE 4] Writing consolidated Zarr warehouse destination path: {self.output_zarr_path}...")
        self.ds.to_zarr(
            self.output_zarr_path,
            mode='w',
            encoding=encoding,
            consolidated=True
        )

        self.ds.close()
        print(f"SUCCESS: Consolidated Multi-Level Icosahedral Zarr compilation finalized at '{self.output_zarr_path}'!\n")


# =====================================================================
# CLI SCRIPT INTERFACE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="High-Speed Multi-Temporal and Multi-Level Icosahedral NetCDF to Compressed Consolidated Zarr Packer."
    )
   #parser.add_argument("-i", "--input", default="starviewergraphcast-grid/global_icosahedral_m4.202[1234]*.nc",
    parser.add_argument("-i", "--input", default="starviewergraphcast-grid/global_icosahedral_m4.202*.nc",
                        help="Input shell glob matching netcdf variables (default: starviewergraphcast-grid/global_icosahedral_m4.*.nc)")
    parser.add_argument("-o", "--output", default="global_icosahedral_m4_3d_heights.zarr",
                        help="Output Zarr path target directory (default: global_icosahedral_m4_3d_heights.zarr)")
    parser.add_argument("-c", "--chunk_size", type=int, default=32,
                        help="Time dimension sequence array chunk size limits (default: 32)")
    parser.add_argument("-l", "--level", type=int, default=3,
                        help="Zstd compressor effort configuration setting [1-9] (default: 3)")

    args = parser.parse_args()

    packer = IcosahedralToZarrPacker(
        file_pattern=args.input,
        output_zarr_path=args.output,
        time_chunk_size=args.chunk_size
    )
    packer.execute_conversion(compression_level=args.level)

if __name__ == "__main__":
    main()
