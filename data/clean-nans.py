import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr

# 1. Configuration Defaults
INPUT_DIR = "./regular_grid"
OUTPUT_DIR = "./regular_grid_cleaned"


def process_file(file_path: str, output_dir: str) -> str:
    """Worker function to inspect a single NetCDF file for NaNs, clean or move it."""
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)

    nan_found = False
    variables_with_nans = []

    # Open dataset with decode_times=False to prevent integer time overflow
    try:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            # Check every data variable (t, u, v, w, q) for NaNs
            for var in ds.data_vars:
                nan_count = ds[var].size - ds[var].count().item()

                if nan_count > 0:
                    nan_found = True
                    variables_with_nans.append((var, nan_count))

            # 2. Handle NaNs if found
            if nan_found:
                # Replace NaNs with 0.0
                ds_cleaned = ds.fillna(0.0)

                # Preserve original NetCDF attributes and metadata
                ds_cleaned.attrs = ds.attrs
                for var in ds.data_vars:
                    ds_cleaned[var].attrs = ds[var].attrs

                # Save clean file to output directory
                ds_cleaned.to_netcdf(output_path)

        # 3. Handle file system moves/deletes after file lock is released
        if nan_found:
            nan_summary = ", ".join([f"{var}: {count}" for var, count in variables_with_nans])
            os.remove(file_path)
            return f"⚠️ {filename} (NaNs found -> [{nan_summary}] -> Cleaned and saved)"
        else:
            os.rename(file_path, output_path)
            return f"✨ {filename} (Clean -> Moved)"

    except Exception as e:
        return f"❌ {filename} (Error processing file: {str(e)})"


def main():
    parser = argparse.ArgumentParser(description="Clean NetCDF files concurrently.")
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        default=INPUT_DIR,
        help=f"Input directory path (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory path (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "-w", "--max_workers",
        type=int,
        default=8,
        help="Number of concurrent worker tasks (e.g. 4, 8, 16. Default: 8)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all target NetCDF files
    file_paths = sorted(glob.glob(os.path.join(args.input_dir, "gfs.*.nc")))

    if not file_paths:
        print(f"❌ No NetCDF files found in {args.input_dir}")
        return

    total_files = len(file_paths)
    print(f"🔍 Found {total_files} files to process in parallel using {args.max_workers} workers.\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_file, file_path, args.output_dir): file_path
            for file_path in file_paths
        }

        for future in as_completed(futures):
            msg = future.result()
            completed += 1
            print(f"[{completed:04d}/{total_files:04d}] {msg}")

    print("\n🎉 All files processed successfully!")


if __name__ == "__main__":
    main()
