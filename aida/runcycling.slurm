#!/bin/bash
#SBATCH --job-name=aida_cycling
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=00:30:00
#SBATCH --output=log.cycling_%J.out
#SBATCH --export=NONE

# 1. Set working directory
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida

# 2. Activate Conda environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

# 3. Define cycling parameters
START_YMD="20250106"
START_HOUR="00"
FCST_HOUR=6

get_analysis_time() {
    local start_ymd=$1
    local start_hour=$2
    local fcst_hour=$3

    # Convert string to a standard "YYYY-MM-DD HH:00:00" format
    local base_time="${start_ymd:0:4}-${start_ymd:4:2}-${start_ymd:6:2} ${start_hour}:00:00"
    echo "base_time: $base_time"

    # Check if the number is negative
    if (( fcst_hour < 0 )); then
       # Convert negative number to positive absolute value for the "ago" syntax
       local abs_hours=$(( fcst_hour * -1 ))
       target_time=$(date -d "$base_time $abs_hours hours ago" +"%Y%m%d %H")
    else
       # Positive shift
       target_time=$(date -d "$base_time $fcst_hour hours" +"%Y%m%d %H")
    fi

    # Assign directly to global variables
    ANALYS_YMD=$(echo "$target_time" | cut -d' ' -f1)
    ANALYS_HOUR=$(echo "$target_time" | cut -d' ' -f2)

    echo "target_time: $target_time"
    echo "ANALYS_YMD: $ANALYS_YMD"
    echo "ANALYS_HOUR: $ANALYS_HOUR"
}

#GNN_CKPT="checkpoints/aida_gnn_surrogate_logstate.pt"
#GNN_CKPT="checkpoints/aida_exp1_smooth004.pt"
GNN_CKPT="checkpoints/aida_gnn_surrogate_logstate.pt"

get_analysis_time ${START_YMD} ${START_HOUR} ${FCST_HOUR}

background=../data/icosahedral_grid/global_icosahedral_m4.${START_YMD}.t${START_HOUR}z.1p00.f006.nc
icosahedral_anal=output/global_icosahedral_m4.${ANALYS_YMD}.t${ANALYS_HOUR}z.1p00.anal.nc
grid_anal=output/reconstructed_aida_analysis_${ANALYS_YMD}.t${ANALYS_HOUR}z.1p00.nc

rm -f ${icosahedral_anal}
rm -f ${grid_anal}

set -x

PYTHONPATH=. python scripts/run_aida_cycling.py \
    --background ${background} \
    --output_file ${icosahedral_anal} \
    --gnn_ckpt ${GNN_CKPT} \
    --graph_path_m4 ../data/graph/icosahedral_edge_index_m4.pt

#   --graph_path_m3 ../data/graph/icosahedral_edge_index_m3.pt \
