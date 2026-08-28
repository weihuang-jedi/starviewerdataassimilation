#!/bin/bash

aidahome=/scratch5/purged/Wei.Huang/src/starviewerdataassimilation
datadir=${aidahome}/data
# 1. Change to working directory
cd ${aidahome}/aida

# 2. Activate Conda environment
source ${aidahome}/svg.env

set -x

yyyymmdd=20260106
analhour=06
res=0p25

# ANALYSIS_FILE=${datadir}/icosahedral-grid/icosahedral_logstate_m6.20260106.t00z.0p25.f006.nc
ANALYSIS_FILE=output/global_icosahedral_m6.${yyyymmdd}.t${analhour}z.${res}.f000.nc
TRUTH_FILE=${datadir}/icosahedral-truth/icosahedral_logstate_m6.${yyyymmdd}.t${analhour}z.${res}.f000.nc
PROILE_FILE=output/verification.${yyyymmdd}.t${analhour}z.csv

python utils/validate_icosahedral.py \
  -i ${ANALYSIS_FILE} \
  -r ${TRUTH_FILE} \
  -o ${PROILE_FILE}

python utils/plot_vertical_profiles.py \
   -i ${PROILE_FILE} \
   -o output/plots/profiles \
   -s

exit 0

