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

GNN_CKPT="checkpoints/aida_gnn_surrogate.pt"
EDGES="../data/graph/icosahedral_edge_index_m4.pt"
BMATRIX="../bmatrix/bmatrix_from_gfs_analysis.nc"
OBS_DIR="../data/conv_2024"
WORK_DIR="./cycling_output_${START_YMD}"
BG_DIR="../data/icosahedral_grid"

rm -rf ${WORK_DIR}/temp_bg_*

# 4. Run AI-DA Cycling orchestrator
python scripts/run_aida_cycling.py \
   --gnn_ckpt ${GNN_CKPT} \
   --graph_path ${EDGES} \
   --data_dir ${BG_DIR} \
   --obs_dir ${OBS_DIR} \
   --bmatrix_path ${BMATRIX} \
   --output_dir ${WORK_DIR} \
   --cycles ${NUM_CYCLES}

exit 0
