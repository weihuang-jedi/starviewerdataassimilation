#!/bin/bash
#SBATCH --job-name=s3_data_transfer
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Allocates 8 CPU cores on the service node
#SBATCH --time=02:00:00
#SBATCH --account=epic
#SBATCH --output=log.s3_transfer_%j.out

set -x

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/data

# Run your new production controller script, matching worker counts to allocated cpus
#python get_data_pipeline.py --start_year 2023 --total_years 1 --res 1p00 --forecast_hour f006 --workers 16

 get_data.sh >& log.data

