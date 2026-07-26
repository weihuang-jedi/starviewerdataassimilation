#!/bin/bash
#SBATCH --job-name=ai_3dvar
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=01:00:00
#SBATCH --output=log.3dvar_%J.out
#SBATCH --export=NONE

# 1. Change to working directory
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida

# 2. Activate Conda environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

# 3. Define specific cycle date and inputs
DATE_STR="20230306"
CYCLE_STR="t18z"

ZARR_BG="../data/icosahedral_2023.zarr"
BMATRIX="../bmatrix/bmatrix_from_gfs_analysis.nc"
EDGES="../data/graph/edge_index_m4.pt"
CONV_OBS="../data/conv_2023/conv.${DATE_STR}.${CYCLE_STR}.nc"
OUT_ANALYSIS="aida_analysis_${DATE_STR}_${CYCLE_STR}.nc"

# 4. Execute B-Matrix Graph 3D-Var Data Assimilation
# Note: Ensure the trailing backslashes '\' are present on every line continuation
python scripts/run_aida_3dvar.py \
    --zarr ${ZARR_BG} \
    --bmatrix ${BMATRIX} \
    --edges ${EDGES} \
    --conv ${CONV_OBS} \
    --output ${OUT_ANALYSIS} \
    --maxiter 30

exit 0

