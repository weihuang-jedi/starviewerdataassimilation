#!/bin/bash
#SBATCH --job-name=s3_data_transfer
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Allocates 8 CPU cores on the service node
#SBATCH --time=02:00:00
#SBATCH --account=epic
#SBATCH --output=log.3dvar_%j.out

set -x

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/traditional

python run_gfs_3dvar_pipeline.py \
    --gfs_file ../data/regular_grid/gfs.20230701.t06z.1p00.f006.nc \
    --bmatrix_file ../bmatrix/bmatrix_from_gfs_analysis.nc \
    --time "2023-07-01" \
    --output gfs.20230701.t12z.1p00.anal.nc \
    --max_iter 30

exit 0

python run_gfs_3dvar_pipeline.py \
    --gfs_file ../data/regular_grid/gfs.20230701.t06z.1p00.f006.nc \
    --bmatrix_file bmatrix_calibrated.nc \
    --time "2023-07-01" \
    --output gfs.20230701.t12z.1p00.anal.nc \
    --max_iter 30
