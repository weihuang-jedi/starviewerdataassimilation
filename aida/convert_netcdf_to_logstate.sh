#!/bin/bash

python scripts/convert_netcdf_to_logstate.py \
  --input_dir ../data/icosahedral_grid \
  --output_dir ../data/icosahedral_grid_logstate \
  --pattern "global_icosahedral_m4*.nc"
