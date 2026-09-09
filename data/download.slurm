#!/bin/bash
#SBATCH --job-name=nnja-obs
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --account=epic
#SBATCH --output=log.fetch_%j.out

set -x

#cd /scratch3/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/data
cd /scratch5/purged/Wei.Huang/src/starviewerdataassimilation/data
source ../svg.env

python utils/fetch_all_nnja_data.py

