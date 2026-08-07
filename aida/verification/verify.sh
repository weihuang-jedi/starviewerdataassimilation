#!/bin/bash

set -e
set -x

# 1. Change to working directory
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida/verification

# 2. Activate Conda Environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

# 3. Define Input and Output Directories
INDIR="output"
REFDIR="../../data/regular_truth"
MASTER_CSV="output/monthly_verification_levels.csv"
MEAN_CSV="output/monthly_mean_verification_levels.csv"
PLOT_DIR="output/plots/monthly"

echo "=========================================================="
echo "[AIDA] Step 1: Processing Batch Monthly Verification Suite"
echo "=========================================================="
python validate_reconstruction.py \
   -i ${INDIR} \
   -r ${REFDIR} \
   -m ${MASTER_CSV} \
   -s ${MEAN_CSV}

echo "=========================================================="
echo "[AIDA] Step 2: Generating Monthly Profiles & 2D Heatmaps"
echo "=========================================================="
python plot_vertical_profiles.py \
   -m ${MASTER_CSV} \
   -s ${MEAN_CSV} \
   -o ${PLOT_DIR}

echo "=========================================================="
echo "[AIDA SUCCESS] Monthly Verification Pipeline Complete!"
echo "Plots saved to: '${PLOT_DIR}'"
echo "=========================================================="

exit 0
