#!/bin/bash
#SBATCH --job-name=aida_training
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=06:00:00
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

# Experiment 1: Relax smoothness slightly to restore sharp synoptic gradients
python scripts/train_aida_surrogate.py \
  --zarr ../data/icosahedral_2023_logstate.zarr \
  --edges ../data/graph/icosahedral_edge_index_m4.pt \
  --epochs 25 \
  --batch_size 4 \
  --lr 0.0003 \
  --tau_min_p -0.2894 \
  --lambda_smooth_p 0.04 \
  --lambda_asym_p 0.5 \
  --checkpoint checkpoints/aida_exp1_smooth004.pt

# Experiment 2: Moderate smoothness + slightly lower state equation weight
python scripts/train_aida_surrogate.py \
  --zarr ../data/icosahedral_2023_logstate.zarr \
  --edges ../data/graph/icosahedral_edge_index_m4.pt \
  --epochs 25 \
  --batch_size 4 \
  --lr 0.0003 \
  --tau_min_p -0.2894 \
  --lambda_smooth_p 0.05 \
  --lambda_asym_p 0.2 \
  --checkpoint checkpoints/aida_exp2_relaxed.pt

set -x

# A. Tuning lambda_smooth_p (Currently 0.08)
# Behavior: Controls spatial smoothness via the graph Laplacian.
# If too high: It over-smoothes real physical gradients (like sharp frontal boundaries and cyclone centers), which will cause ACC to plateau or drop.
# If too low: Checkerboard noise creeps back in.
# Next Action: Run a 3-point sweep around 0.08:
# Try 0.04: Check if relaxing the penalty sharpens pressure gradients and increases ACC.
# Try 0.12: Test if the field is still slightly noisy.

python scripts/train_aida_surrogate.py \
  --zarr ../data/icosahedral_2023_logstate.zarr \
  --edges ../data/graph/icosahedral_edge_index_m4.pt \
  --checkpoint checkpoints/aida_gnn_surrogate_logstate.pt \
  --epochs 25 \
  --batch_size 4 \
  --lr 0.0003 \
  --tau_min_p -0.2894 --lambda_smooth_p 0.10 --lambda_asym_p 0.25

#if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
#   sbatch runcycling.slurm
#fi

exit 0

# B. Tuning lambda_asym_p (Currently 0.5)
# Behavior: Acts as a barrier against non-physical extreme low pressure.
# Next Action: Look at your diagnostic log output for AsymP (or loss_asym_p):
# If AsymP drops to $\sim 0.0$ early in training:
# The network is no longer trying to predict unphysical low values ($\le \tau_{\text{min\_p}}$).
# You can keep lambda_asym_p = 0.5 or even reduce it to 0.2 to avoid unnecessary gradient pressure during late epochs.
# If AsymP remains high throughout training: The model is still pushing against the physical barrier. Keep 0.5 or increase to 1.0.

# Notice the trailing backslash '\' on the python line below:
python scripts/train_aida_surrogate.py \
  --zarr ../data/icosahedral_2023_logstate.zarr \
  --edges ../data/graph/icosahedral_edge_index_m4.pt \
  --checkpoint checkpoints/aida_gnn_surrogate_logstate.pt \
  --epochs 25 \
  --batch_size 4 \
  --lr 0.0003 \
  --tau_min_p -0.2894 --lambda_smooth_p 0.08 --lambda_asym_p 0.5

exit 0

    parser.add_argument("-z", "--zarr", default="../data/icosahedral_2023_logstate.zarr", help="Path to input log-state .zarr store")
    parser.add_argument("-g", "--edges", default="../data/graph/icosahedral_edge_index_m4.pt", help="Path to PyTorch edge_index tensor")
    parser.add_argument("-c", "--checkpoint", default="checkpoints/aida_gnn_surrogate_logstate.pt", help="Output model path")
    parser.add_argument("-e", "--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--tau_min_p", type=float, default=-3.5, help="Threshold below which asymmetric penalty activates, a tuned value: -0.2894")
    parser.add_argument("--lambda_smooth_p", type=float, default=0.08, help="Pressure smoothness loss weight")
    parser.add_argument("--lambda_asym_p", type=float, default=0.5, help="Asymmetric pressure penalty weight")

