#!/bin/bash

set -x

# 1. Change to working directory
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida

# 2. Activate Conda environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

yyyymmdd=20240106
analhour=06

origanalysis=cycling_output_${yyyymmdd}/aida_analysis_${yyyymmdd}_t${analhour}z.nc
gridanalysis=cycling_output_${yyyymmdd}/reconstructed_aida_analysis_${yyyymmdd}_t${analhour}z.nc

#python scripts/verify_aida_logstate.py \
#   -a cycling_output_20240106/aida_analysis_cycle_01.nc \
#   -t ../data/nc/truth_t06z.nc

if [ ! -f ${gridanalysis} ]; then
   python icosahedral2regular.py \
      -i ${origanalysis} \
      -o ${gridanalysis} \
      -r 1.0
fi

python plot_3dvar_matrix.py \
   --input ${gridanalysis} \
   --output aida_increment_matrix_${yyyymmdd}T${analhour}.png

python validate_reconstruction.py \
   -i ${gridanalysis} \
   -r ../data/regular_truth/gfs.${yyyymmdd}.t${analhour}z.1p00.f000.nc \
   -o cycling_output_${yyyymmdd}/verification_levels_t${analhour}z.csv

exit 0

