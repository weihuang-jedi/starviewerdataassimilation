#!/bin/bash
#SBATCH --job-name=aida_cycling
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=00:30:00
#SBATCH --output=log.cycling_%J.out
#SBATCH --export=NONE

# 1. Set working directory
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida

# 2. Activate Conda environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

# 3. Define cycling parameters
START_YMD="20240106"
START_HOUR="00"
START_DATE="${START_YMD}_${START_HOUR}"
NUM_CYCLES=4  # Runs 4x 6-hour cycles (24 hours total)

GNN_CKPT="checkpoints/aida_gnn_surrogate_logstate.pt"
WORK_DIR="./cycling_output_${START_YMD}"

background=../data/icosahedral_grid_logstate/global_icosahedral_m4.20240106.t00z.1p00.f006.nc

rm -rf ${WORK_DIR}/temp_bg_*

set -x

PYTHONPATH=. python scripts/run_aida_cycling.py \
  --background ../data/icosahedral_grid_logstate/global_icosahedral_m4.20240106.t00z.1p00.f006.nc \
  --output_file output/global_icosahedral_m4.20240106.t06z.1p00.anal.nc \
  --gnn_ckpt checkpoints/aida_gnn_surrogate_logstate.pt \
  --graph_path_m4 ../data/graph/icosahedral_edge_index_m4.pt \
  --graph_path_m3 ../data/graph/icosahedral_edge_index_m3.pt

exit 0

PYTHONPATH=. python scripts/run_aida_cycling.py \
  --background ../data/icosahedral_grid_logstate/global_icosahedral_m4.20240106.t00z.1p00.f006.nc \
  --output_dir output \
  --gnn_ckpt checkpoints/aida_gnn_surrogate_logstate.pt \
  --graph_path_m4 ../data/graph/icosahedral_edge_index_m4.pt \
  --graph_path_m3 ../data/graph/icosahedral_edge_index_m3.pt
