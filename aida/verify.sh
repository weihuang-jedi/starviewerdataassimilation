#!/bin/bash

aidahome=/scratch5/purged/Wei.Huang/src/starviewerdataassimilation
datadir=/scratch5/purged/Wei.Huang/src/starviewerdataassimilation/data
# 1. Change to working directory
cd ${aidahome}/aida

# 2. Activate Conda environment
source ${aidahome}/svg.env

yyyymmdd=20260106
analhour=06
res=0p25

set -x

icosahedral_analysis=output/global_icosahedral_m6.${yyyymmdd}.t${analhour}z.${res}.f000.nc
gridanalysis=output/reconstructed_aida_analysis_${yyyymmdd}.t${analhour}z.${res}.nc

if [ ! -f ${gridanalysis} ]; then
   python utils/icosahedral2regular.py \
      -i ${icosahedral_analysis} \
      -o ${gridanalysis} \
      -g ${datadir}/terrain-regular-grid/gfs.${yyyymmdd}.t${analhour}z.${res}.f000.nc \
      -r 0.25
fi

#python plot_3dvar_matrix.py \
#   --input ${gridanalysis} \
#   --output aida_increment_matrix_${yyyymmdd}T${analhour}.png

python utils/validate_reconstruction.py \
   -i ${gridanalysis} \
   -r ${datadir}/terrain-regular-grid/gfs.${yyyymmdd}.t${analhour}z.${res}.f000.nc \
   -o output/verification_levels.${yyyymmdd}.t${analhour}z.csv

python utils/plot_vertical_profiles.py \
   -i output/verification_levels.${yyyymmdd}.t${analhour}z.csv \
   -o output/plots/profiles

exit 0

