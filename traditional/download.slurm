#!/bin/bash
#SBATCH --job-name=s3_data_transfer
#SBATCH --partition=u1-service
##SBATCH --partition=u1-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Allocates 8 CPU cores on the service node
#SBATCH --time=00:30:00
#SBATCH --account=epic
#SBATCH --output=log.3dvar_%j.out

set -x

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/traditional

source ../svg.env

python fetch_nnja_obs.py --datetime 2023-07-01T12:00:00 --out_conv conv_adpupa_NC002001.nc --out_amsua amsua_NC021023.nc
