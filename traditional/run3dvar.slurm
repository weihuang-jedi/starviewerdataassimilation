#!/bin/bash
#SBATCH --job-name=s3_data_transfer
##SBATCH --partition=u1-service
#SBATCH --partition=u1-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Allocates 8 CPU cores on the service node
#SBATCH --time=01:00:00
#SBATCH --account=epic
#SBATCH --output=log.3dvar_%j.out

set -x

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/traditional

python run_gfs_3dvar_pipeline.py \
    --bg ../data/regular_grid/gfs.20230701.t06z.1p00.f006.nc \
    --amsua amsua_NC021023.nc \
    --conv conv_adpupa_NC002001.nc \
    --output gfs.20230701.t12z.1p00.anal.nc \
    --maxiter 30

exit 0
    --bmatrix_file ../bmatrix/bmatrix_from_gfs_analysis.nc \
    --time "2023-07-01" \

usage: run_gfs_3dvar_pipeline.py [-h] --bg BG [--conv CONV] [--amsua AMSUA]
                                 [--output OUTPUT] [--maxiter MAXITER]

Run 5-Var GFS 3D-Var Data Assimilation Pipeline

options:
  -h, --help         show this help message and exit
  --bg BG            Path to Background NetCDF file
  --conv CONV        Path to Conventional Obs NetCDF file
  --amsua AMSUA      Path to AMSU-A Radiance Obs NetCDF file
  --output OUTPUT    Output Analysis NetCDF path
  --maxiter MAXITER  Maximum L-BFGS-B iterations
