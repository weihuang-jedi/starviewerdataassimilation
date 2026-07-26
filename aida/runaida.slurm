#!/bin/bash
#SBATCH --job-name=training
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

# Notice the trailing backslash '\' on the python line below:
python scripts/train_aida_surrogate.py \
    --zarr ../data/icosahedral_2023.zarr \
    --bmatrix ../bmatrix/bmatrix_from_gfs_analysis.nc \
    --edges ../data/graph/edge_index_m4.pt \
    --out_ckpt ../checkpoints/aida_gnn_v1.pt \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-4

exit 0
