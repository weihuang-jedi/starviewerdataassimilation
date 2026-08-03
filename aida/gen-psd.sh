#!/bin/bash

set -x

python scripts/eval_psd_diagnostics.py \
   --checkpoint checkpoints/aida_gnn_surrogate_logstate.pt \
   --output diagnostics/psd_pressure_spectrum.png

