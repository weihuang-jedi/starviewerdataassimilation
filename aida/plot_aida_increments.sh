#!/bin/bash

set -x

#  -b ../data/regular_truth/gfs.20240106.t06z.1p00.f000.nc \

python utils/plot_aida_increments.py \
   -b ../data/icosahedral-grid/icosahedral_logstate_m6.20260102.t00z.0p25.f006.nc \
   -a output/global_icosahedral_m6.20260106.t06z.0p25.f000.nc \
   -idx 10 -o output/plots -s

