#!/bin/bash

set -x

python scripts/convert_zarr_logstate.py \
  --input ../data/icosahedral_2023.zarr \
  --output ../data/icosahedral_2023_logstate.zarr
