#!/bin/bash
#SBATCH --job-name=nnja-obs
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=epic
#SBATCH --output=log.fetch_%j.out

set -x

cd /scratch5/purged/Wei.Huang/src/starviewerdataassimilation/data
source ../svg.env

#python fetch_nnja_obs.py
#python fetch_nnja_amsua_iasi_obs.py
#python fetch_nnja_amsua_iasi_hms.py
python fetch_conv_amsua_iasi_hms_atms.py
