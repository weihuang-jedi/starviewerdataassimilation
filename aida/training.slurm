#!/bin/bash
#SBATCH --job-name=aida_training
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=00:30:00
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

time python scripts/train_aida_surrogate.py \
  --zarr ../data/icosahedral_2023_logstate.zarr \
  --edges ../data/graph/icosahedral_edge_index_m4.pt \
  --checkpoint checkpoints/aida_gnn_surrogate_logstate.pt \
  --epochs 25 \
  --batch_size 4 \
  --lr 0.0003 \
  --tau_min_p -0.2894 \
  --lambda_smooth_p 0.15 \
  --weight_grad_state 0.20 \
  --lambda_asym_p 0.25

if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
   sbatch runcycling.slurm
fi

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

