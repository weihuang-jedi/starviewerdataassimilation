#!/bin/bash

set -x

background=../data/icosahedral-truth/icosahedral_logstate_m6.20260106.t06z.0p25.f000.nc
#background=../data/icosahedral-grid/icosahedral_logstate_m6.20260102.t00z.0p25.f006.nc

python utils/plot_aida_increments.py \
   -b ${background} \
   -a output/global_icosahedral_m6.20260106.t06z.0p25.f000.nc \
   -idx 10 -o output/plots -s

