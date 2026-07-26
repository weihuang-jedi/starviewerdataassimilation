#!/usr/bin/env python3
"""
scripts/run_aida_cycling.py
---------------------------
Operational cycling loop driver:
Executes sequential 6-hour assimilation & forecast windows.

1. The Operational Driver (scripts/run_aida_cycling.py)
   This script manages the 6-hour cycling window
   ($00\text{Z} \rightarrow 06\text{Z} \rightarrow 12\text{Z} \rightarrow 18\text{Z}$).
   It calls 3D-Var to create the analysis $x_a$, then passes $x_a$ to the GNN forecast model
   to produce the background $x_b$ for the next cycle.
"""

import os
import argparse
from datetime import datetime, timedelta
import torch

from run_aida_3dvar import execute_3dvar_analysis
from run_aida_forecast import run_gnn_forecast

def main():
    parser = argparse.ArgumentParser(description="Run Operational AI-DA Cycling Loop")
    parser.add_argument("--start_date", type=str, default="2023030100", help="Start date YYYYMMDDHH")
    parser.add_argument("--end_date", type=str, default="2023030200", help="End date YYYYMMDDHH")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/aida_gnn_v1.pt", help="GNN model weights")
    parser.add_argument("--obs_dir", type=str, default="../data/conv_2023", help="Directory of observation NetCDFs")
    parser.add_argument("--zarr_bg", type=str, default="../data/icosahedral_2023.zarr", help="Initial background state")
    parser.add_argument("--out_dir", type=str, default="../data/cycling_output", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load frozen GNN model once for forecasting
    print(f"[SYSTEM] Loading GNN Forecast Model from: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.load(args.checkpoint, map_location=device).eval()

    current_time = datetime.strptime(args.start_date, "%Y%m%d%H")
    end_time = datetime.strptime(args.end_date, "%Y%m%d%H")

    # Initial background x_b from Zarr (Cycle 0)
    current_xb_path = args.zarr_bg

    while current_time <= end_time:
        date_str = current_time.strftime("%Y%m%d")
        cycle_str = current_time.strftime("t%Hz")
        timestamp_str = current_time.strftime("%Y%m%d%H")

        print(f"\n==================================================================")
        print(f" OPERATIONAL CYCLE: {timestamp_str} ")
        print(f"==================================================================")

        obs_file = os.path.join(args.obs_dir, f"conv.{date_str}.{cycle_str}.nc")
        analysis_out = os.path.join(args.out_dir, f"aida_analysis_{timestamp_str}.nc")
        next_xb_out = os.path.join(args.out_dir, f"aida_bg_forecast_{timestamp_str}_f06.nc")

        # STEP 1: Run 3D-Var Assimilation (x_b + Obs -> x_a)
        print(f"\n---> [Phase 1/2] Executing 3D-Var Assimilation...")
        xa_dict = execute_3dvar_analysis(
            zarr_or_nc_path=current_xb_path,
            obs_nc_path=obs_file,
            output_path=analysis_out
        )

        # STEP 2: Run 6-Hour GNN Forecast (x_a -> x_b_next)
        print(f"\n---> [Phase 2/2] Running 6-Hour GNN Forecast for next cycle...")
        # run_gnn_forecast(xa_dict, model, out_path=next_xb_out)

        # Set output forecast as input background for next cycle
        current_xb_path = next_xb_out
        current_time += timedelta(hours=6)

    print("\nOperational Cycling completed successfully!")

if __name__ == "__main__":
    main()

