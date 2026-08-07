#!/bin/bash
#SBATCH --job-name=aida_training
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=03:45:00
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
time python -u scripts/train_aida_surrogate.py --config configs/config.yaml

if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
   runcycling.slurm
fi

exit 0

