#!/bin/bash
#SBATCH --job-name=aida_analysis
#SBATCH --account=epic
#SBATCH --mem=192G
#SBATCH --nodes=1
#SBATCH --partition=u1-compute
##SBATCH --partition=u1-h100
##SBATCH --gres=gpu:h100:1
##SBATCH --qos=gpuwf
#SBATCH --time=02:30:00
#SBATCH --output=log.analysis_%J.out
#SBATCH --export=NONE

aidahome=/scratch5/purged/Wei.Huang/src/starviewerdataassimilation
# 1. Set working directory
cd ${aidahome}/aida

# 2. Activate Conda environment
source ${aidahome}/svg.env

# 3. Define cycling parameters
START_YMD="20260106"
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

GNN_CKPT="checkpoints/aida_gnn_surrogate_latest_batch.pt"
#GNN_CKPT="checkpoints/aida_gnn_surrogate_logstate_epoch_002.pt"
#GNN_CKPT="checkpoints/aida_gnn_surrogate_logstate_epoch_004.pt"

get_analysis_time ${START_YMD} ${START_HOUR} ${FCST_HOUR}

background=../data/icosahedral-grid/icosahedral_logstate_m6.${START_YMD}.t${START_HOUR}z.0p25.f006.nc
icosahedral_anal=output/global_icosahedral_m6.${ANALYS_YMD}.t${ANALYS_HOUR}z.0p25.f000.nc

rm -f ${icosahedral_anal}
rm -f ${grid_anal}
rm -f output/*

set -x

PYTHONPATH=. python scripts/run_aida_analysis.py \
    --checkpoint ${GNN_CKPT} \
    --background ${background} \
    --edges=../graph/graph-grid/icosahedral_edge_index_m6.pt \
    --output ${icosahedral_anal}

verify.sh

exit 0
