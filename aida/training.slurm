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

# Notice the trailing backslash '\' on the python line below:
python scripts/train_aida_surrogate.py \
    --data_dir ../data/icosahedral_grid \
    --edges ../data/graph/edge_index_m4.pt \
    --bmatrix ../bmatrix/bmatrix_from_gfs_analysis.nc \
    --epochs 20 \
    --out_ckpt checkpoints/aida_gnn_v1.pt \
    --batch_size 4 \
    --lr 1e-4

exit 0
    parser.add_argument("--data_dir", type=str, default="../data/icosahedral_grid")
    parser.add_argument("--edges", type=str, default="../data/graph/edge_index_m4.pt")
    parser.add_argument("--bmatrix", type=str, default="../bmatrix/bmatrix_from_gfs_analysis.nc")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--out_ckpt", type=str, default="../checkpoints/aida_gnn_v1.pt")

