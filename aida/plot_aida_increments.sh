#!/bin/bash

set -x

#  -b ../data/regular_truth/gfs.20240106.t06z.1p00.f000.nc \

python utils/plot_aida_increments.py \
   -b ../data/regular_grid/gfs.20240106.t00z.1p00.f006.nc \
   -a output/reconstructed_aida_analysis_20240106.t06z.1p00.nc \
   -idx 10 -o output/plots

