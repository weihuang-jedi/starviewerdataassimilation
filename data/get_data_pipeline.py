#!/usr/bin/env python3
import os
import sys
import datetime
import calendar
import argparse
import subprocess
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ProcessPoolExecutor, as_completed

class GFSPipelineManager:
    """
    Manages concurrent downloading, height interpolation, and icosahedral mapping 
    of GFS data fields from the NOAA Public Dataset on AWS S3.
    """
    def __init__(self, start_year: int, total_years: int, resolution: str, forecast_hour: str, max_workers: int = 4):
        self.start_year = start_year
        self.total_years = total_years
        self.forecast_hour = forecast_hour
        self.resolution = resolution
        self.max_workers = max_workers
        
        self.bucket_name = "noaa-gfs-bdp-pds"
        # Configure anonymous S3 Client to bypass credentials safely
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        
        # Ensure your local workspace directory architecture is setup safely
        os.makedirs("regular_grid", exist_ok=True)
        os.makedirs("icosahedral_grid", exist_ok=True)

    def _generate_tasks(self):
        """Generates all potential task definitions across the targeted date matrix range."""
        tasks = []
        hours = ["00", "06", "12", "18"]
        
        for n in range(self.total_years + 1):
            current_year = self.start_year + n
            for month in range(1, 13):
                # Calculate absolute maximum days dynamically per calendar month
                _, max_days = calendar.monthrange(current_year, month)
                for day in range(1, max_days + 1):
                    
                    # Pad date metrics safely
                    yyyy = f"{current_year}"
                    mm = f"{month:02d}"
                    dd = f"{day:02d}"
                    
                    for hh in hours:
                        task = {
                            "yyyy": yyyy, "mm": mm, "dd": dd, "hh": hh,
                            "s3dir": f"gfs.{yyyy}{mm}{dd}",
                            "iflnm": f"gfs.t{hh}z.pgrb2.{self.resolution}.{self.forecast_hour}",
                            "ncflnm": f"regular_grid/gfs.{yyyy}{mm}{dd}.t{hh}z.{self.resolution}.{self.forecast_hour}.nc",
                            "mlgridflnm": f"icosahedral_grid/global_icosahedral_m4.{yyyy}{mm}{dd}.t{hh}z.{self.resolution}.{self.forecast_hour}.nc"
                        }
                        tasks.append(task)
        return tasks

    @staticmethod
    def process_single_task(task: dict, bucket_name: str, resolution: str) -> str:
        """
        Static execution node running individual pipeline steps. Must be static 
        to pass cleanly across process bounds via the ProcessPoolExecutor.
        """
        # Unique naming to isolate concurrent local download tasks cleanly
        local_grib = f"tmp_{task['yyyy']}{task['mm']}{task['dd']}_{task['hh']}_{task['iflnm']}"
        local_grib_idx = f"{local_grib}.5b7b6.idx"
        
        # Skip everything if final downstream node file is already fully generated
        if os.path.exists(task['mlgridflnm']):
            return f"[SKIP] Final target exists: {task['mlgridflnm']}"

        try:
            # Step 1: Handle GRIB data file layer checks
            if not os.path.exists(task['ncflnm']):
                if not os.path.exists(local_grib):
                    # Direct Python download via S3 Client (replaces aws s3 cp CLI command)
                    # before or at 20210322T12
                    # s3_key = f"{task['s3dir']}/{task['hh']}/{task['iflnm']}"
                    # after 20210322T12
                    s3_key = f"{task['s3dir']}/{task['hh']}/atmos/{task['iflnm']}"
                    s3_anon = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                    
                    print(f"[DOWNLOADING] s3://{bucket_name}/{s3_key} -> {local_grib}")
                    s3_anon.download_file(bucket_name, s3_key, local_grib)

                # Step 2: Run Height Interpolation Engine Script Pass
                print(f"[INTERPOLATING HEIGHTS] Processing: {local_grib}")
                cmd_height = ["python", "interpolate_to_heights.py", "-i", local_grib, "-o", task['ncflnm']]
                subprocess.run(cmd_height, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

                # Active local cleanup of raw data sheets
                if os.path.exists(local_grib):
                    os.remove(local_grib)
                if os.path.exists(local_grib_idx):
                    os.remove(local_grib_idx)

            # Step 3: Run Icosahedral Grid Remapping Module Pass
            print(f"[REMAPPING ICOSAHEDRAL] Generating: {task['mlgridflnm']}")
            cmd_icos = [
                "python", "interpolate2icosahedral.py",
                "--input", task['ncflnm'],
                "--mesh", "../graph/starviewergraphcast-grid/global_icosahedral_mesh_m4.nc",
                "--output", task['mlgridflnm']
            ]
            subprocess.run(cmd_icos, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return f"[SUCCESS] Fully compiled target: {task['mlgridflnm']}"

        except subprocess.CalledProcessError as e:
            # Cleanup broken transient downloads on error
            if os.path.exists(local_grib):
                os.remove(local_grib)
            return f"[FAILURE] Processing Error on {task['yyyy']}-{task['mm']}-{task['dd']} {task['hh']}Z: {e.stderr.decode().strip()}"
        except Exception as e:
            if os.path.exists(local_grib):
                os.remove(local_grib)
            return f"[FAILURE] Unexpected Error: {str(e)}"

    def run_pipeline(self):
        """Launches the execution engine matrix pool across the thread cores."""
        all_tasks = self._generate_tasks()
        print(f"Initialized GFS Pipeline. Discovered {len(all_tasks)} potential target cycles.")
        print(f"Engaging execution matrix with a pool allocation limit of {self.max_workers} processes.\n")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all targets to the asynchronous loop matrix
            futures = {
                executor.submit(self.process_single_task, task, self.bucket_name, self.resolution): task 
                for task in all_tasks
            }

            # Gather runtime tracking feedback live as pools complete processing loops
            for future in as_completed(futures):
                result_message = future.result()
                print(result_message)

def main():
    parser = argparse.ArgumentParser(description="Concurrent Production Python Controller Engine for NOAA GFS Datasets.")
    parser.add_argument("--start_year", type=int, default=2023, help="Starting year coordinate.")
    parser.add_argument("--total_years", type=int, default=1, help="Number of years to add to starting point.")
    parser.add_argument("--res", type=str, default="1p00", choices=["1p00", "0p50", "0p25"], help="GFS Resolution sheet grid sizing.")
    parser.add_argument("--forecast_hour", type=str, default="f006", choices=["f006", "f012", "f024", ...], help="GFS forecast time string.")
    parser.add_argument("--workers", type=int, default=4, help="Maximum concurrent multi-process worker tracks to assign.")

    args = parser.parse_args()

    pipeline = GFSPipelineManager(
        start_year=args.start_year,
        total_years=args.total_years,
        forecast_hour=args.forecast_hour,
        resolution=args.res,
        max_workers=args.workers
    )
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
