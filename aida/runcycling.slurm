#!/bin/bash
#SBATCH --job-name=aida_cycling
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=03:00:00
#SBATCH --output=log.cycling_%J.out
#SBATCH --export=NONE

# 1. Set working directory
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida

# 2. Activate Conda environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

# 3. Define cycling parameters
START_DATE="20230306_00"
NUM_CYCLES=4  # Runs 4x 6-hour cycles (24 hours total)

GNN_CKPT="../checkpoints/aida_gnn_v1.pt"
EDGES="../data/graph/edge_index_m4.pt"
ZARR_INIT="../data/icosahedral_2023.zarr"
BMATRIX="../bmatrix/bmatrix_from_gfs_analysis.nc"
OBS_DIR="../data/conv_2023"
WORK_DIR="./cycling_output_20230306"


rm -rf ./cycling_output_20230306/temp_bg_*
# 4. Run AI-DA Cycling orchestrator
python scripts/run_aida_cycling.py \
    --start_date ${START_DATE} \
    --num_cycles ${NUM_CYCLES} \
    --gnn_ckpt ${GNN_CKPT} \
    --edges ${EDGES} \
    --zarr_init ${ZARR_INIT} \
    --bmatrix ${BMATRIX} \
    --obs_dir ${OBS_DIR} \
    --work_dir ${WORK_DIR}

exit 0
