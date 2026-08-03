#!/bin/bash
#SBATCH --job-name=aida_training
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
python -u scripts/train_aida_surrogate.py \
    --zarr ../data/icosahedral_2023_logstate.zarr \
    --edges ../data/graph/icosahedral_edge_index_m4.pt \
    --checkpoint checkpoints/aida_gnn_surrogate_logstate.pt \
    --epochs 35 \
    --tau_min_p -0.25 \
    --lambda-laplacian-p 0.18 \
    --weight-grad-state 0.20 \
    --weight-state-eq 0.10 \
    --weight-q-log 0.15 \
    --weight-joint-bias 0.05

if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
   runcycling.slurm
fi

exit 0

# 2. Run single-step cycling inference
python -u scripts/run_aida_cycling.py \
    --background ../data/background_state.nc \
    --output_file ../data/analysis_state.nc \
    --gnn_ckpt checkpoints/aida_gnn_surrogate_logstate.pt \
    --graph_path_m4 ../data/graph/icosahedral_edge_index_m4.pt

