#!/bin/bash

set -x

# 1. Change to working directory
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida

# 2. Activate Conda environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

yyyymmdd=20230306
analhour=06

origanalysis=cycling_output_${yyyymmdd}/aida_analysis_${yyyymmdd}_t${analhour}z.nc
gridanalysis=cycling_output_${yyyymmdd}/reconstructed_aida_analysis_${yyyymmdd}_t${analhour}z.nc

if [ ! -f ${gridanalysis} ]; then
   python icosahedral2regular.py \
      -i ${origanalysis} \
      -o ${gridanalysis} \
      -r 1.0
fi

python plot_3dvar_matrix.py \
   --input ${gridanalysis} \
   --output aida_increment_matrix_${yyyymmdd}.png

exit 0

