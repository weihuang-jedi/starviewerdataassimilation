#!/bin/bash

set -x

# 1. Change to working directory
#cd /scratch3/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida
# cd /scratch5/purged/Wei.Huang/src/starviewerdataassimilation/aida

# 2. Activate Conda environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

yyyymmdd=20250106
analhour=06

icosahedral_analysis=output/global_icosahedral_m4.${yyyymmdd}.t${analhour}z.1p00.anal.nc
gridanalysis=output/reconstructed_aida_analysis_${yyyymmdd}.t${analhour}z.1p00.nc

if [ ! -f ${gridanalysis} ]; then
   python icosahedral2regular.py \
      -i ${icosahedral_analysis} \
      -o ${gridanalysis} \
      -g /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/data/regular_truth/gfs.${yyyymmdd}.t${analhour}z.1p00.f000.nc \
      -r 1.0
fi

#python plot_3dvar_matrix.py \
#   --input ${gridanalysis} \
#   --output aida_increment_matrix_${yyyymmdd}T${analhour}.png

python validate_reconstruction.py \
   -i ${gridanalysis} \
   -r /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/data/regular_truth/gfs.${yyyymmdd}.t${analhour}z.1p00.f000.nc \
   -o output/verification_levels.t${analhour}z.csv

python utils/plot_vertical_profiles.py \
   -i output/verification_levels.t06z.csv \
   -o output/plots/profiles

exit 0

