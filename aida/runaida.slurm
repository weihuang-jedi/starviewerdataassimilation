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

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida

python run_aida_graph_3dvar.py \
    --zarr ../data/icosahedral_2023.zarr \
    --edges ../data/graph \
    --conv CONV        Path to conventional observations
    --output gfs.20230701.t12z.1p00.aida.nc \
    --maxiter 30

exit 0

(anemoi) [Wei.Huang@ufe02 aida]$ python run_aida_graph_3dvar.py --help
usage: run_aida_graph_3dvar.py [-h] [--zarr ZARR] [--edges EDGES] [--conv CONV] [--output OUTPUT] [--maxiter MAXITER]

Run AI-DA Graph-based 3D-Var Engine on Icosahedral Mesh

options:
  -h, --help         show this help message and exit
  --zarr ZARR        Path to background Zarr
  --edges EDGES      Path to PyTorch graph edges
  --conv CONV        Path to conventional observations
  --output OUTPUT    Output NetCDF analysis file
  --maxiter MAXITER  Max optimization iterations

