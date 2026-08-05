#!/bin/bash
#SBATCH --job-name=aida_forecast
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=00:45:00
#SBATCH --output=log.training_%J.out
#SBATCH --export=NONE

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
   n=0
   while [ -f checkpoints/aida_gnn_surrogate_logstate_${n}.pt ]; do
      n=$(( n + 1 ))
   done
   mv checkpoints/aida_gnn_surrogate_logstate.pt checkpoints/aida_gnn_surrogate_logstate_${n}.pt
fi

set -x

# 1. Train surrogate
time \
python -u scripts/evaluate_aida_rollout.py \
  --zarr ../data/icosahedral_2023_logstate.zarr \
  --edges ../data/graph/icosahedral_edge_index_m4.pt \
  --checkpoint checkpoints/aida_gnn_surrogate_logstate_12.pt \
  --rollout-steps 28 \
  --num-samples 5 \
  --output-plot output/aida_ar_rollout_7day.png

