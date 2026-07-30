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

set -x

# Notice the trailing backslash '\' on the python line below:
python scripts/train_aida_surrogate.py \
    --zarr ../data/icosahedral_2023.zarr \
    --edges ../data/graph/icosahedral_edge_index_m4.pt \
    --checkpoint checkpoints/aida_gnn_surrogate.pt \
    --epochs 25 \
    --batch_size 4 \
    --lr 0.0003

exit 0

