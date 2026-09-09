#!/bin/bash
#SBATCH --job-name=aida_training
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=15:00:00
#SBATCH --output=log.training_%J.out
#SBATCH --export=NONE

aidahome=/scratch5/purged/Wei.Huang/src/starviewerdataassimilation
source ${aidahome}/svg.env
cd ${aidahome}/aida

if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
   n=0
   while [ -f checkpoints/aida_gnn_surrogate_logstate_${n}.pt ]; do
      n=$(( n + 1 ))
   done
   mv checkpoints/aida_gnn_surrogate_logstate.pt checkpoints/aida_gnn_surrogate_logstate_${n}.pt
fi

set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Train surrogate
time python -u scripts/train_aida_surrogate.py --config configs/config.yaml

#if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
#   runcycling.slurm
#fi

exit 0

