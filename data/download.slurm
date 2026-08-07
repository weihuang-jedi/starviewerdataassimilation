#!/bin/bash
#SBATCH --job-name=nnja-obs
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Allocates 8 CPU cores on the service node
#SBATCH --time=02:00:00
#SBATCH --account=epic
#SBATCH --output=log.fetch_%j.out

set -x

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida/data

source ../svg.env

#python fetch_nnja_obs.py
python fetch_nnja_amsua_iasi_obs.py
