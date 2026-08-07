#!/bin/bash
#SBATCH --job-name=aida_verification
#SBATCH --account=epic
#SBATCH --mem=192G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=u1-compute
#SBATCH --time=04:30:00
#SBATCH --output=log.verification_%J.out
#SBATCH --export=NONE

# 1. Set working directory
cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/aida/verification

# 2. Activate Conda environment
source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

# Define the processing logic for a single analysis step
process_analysis() {
   local CUR_YMD=$1
   local CUR_HOUR=$2
   local ANALYS_YMD=$3
   local ANALYS_HOUR=$4
   local GNN_CKPT=$5

   local background=../../data/icosahedral_grid/global_icosahedral_m4.${CUR_YMD}.t${CUR_HOUR}z.1p00.f006.nc
   local icosahedral_anal=output/global_icosahedral_m4.${ANALYS_YMD}.t${ANALYS_HOUR}z.1p00.anal.nc
   local grid_analysis=output/reconstructed_aida_analysis_${ANALYS_YMD}.t${ANALYS_HOUR}z.1p00.nc

   # Clean up prior outputs if they exist
   rm -f "${icosahedral_anal}" "${grid_analysis}"

   # Run Python GNN cycling script
   PYTHONPATH=. python ../scripts/run_aida_cycling.py \
       --background "${background}" \
       --output_file "${icosahedral_anal}" \
       --gnn_ckpt "${GNN_CKPT}" \
       --graph_path_m4 ../data/graph/icosahedral_edge_index_m4.pt

   # Run interpolation to regular grid
   # Note: Fixed original variable names ($yyyymmdd and $analhour changed to match inputs)
   python ../icosahedral2regular.py \
         -i "${icosahedral_anal}" \
         -o "${grid_analysis}" \
         -g "../../data/regular_truth/gfs.${ANALYS_YMD}.t${ANALYS_HOUR}z.1p00.f000.nc" \
         -r 1.0
}

# Export function and static variable for xargs subshells
export -f process_analysis
export GNN_CKPT="aida_gnn_surrogate_logstate.pt"

# Function to pre-calculate dates and stream arguments to xargs
generate_tasks() {
   local START_YMD="20250101"
   local START_HOUR="00"
   local FCST_HOUR=6
   local total_analysis=$(( 4 * 31 ))

   local CUR_YMD=${START_YMD}
   local CUR_HOUR=${START_HOUR}

   local n=0
   while [ $n -lt $total_analysis ]
   do
      # Calculate targets using GNU date utility
      local base_time="${CUR_YMD:0:4}-${CUR_YMD:4:2}-${CUR_YMD:6:2} ${CUR_HOUR}:00:00"
      local target_time
      
      if (( FCST_HOUR < 0 )); then
         local abs_hours=$(( FCST_HOUR * -1 ))
         target_time=$(date -d "$base_time $abs_hours hours ago" +"%Y%m%d %H")
      else
         target_time=$(date -d "$base_time $FCST_HOUR hours" +"%Y%m%d %H")
      fi

      local ANALYS_YMD
      local ANALYS_HOUR
      ANALYS_YMD=$(echo "$target_time" | cut -d' ' -f1)
      ANALYS_HOUR=$(echo "$target_time" | cut -d' ' -f2)

      # Output arguments for 1 task per line
      echo "${CUR_YMD} ${CUR_HOUR} ${ANALYS_YMD} ${ANALYS_HOUR} ${GNN_CKPT}"

      # Sequence to the next step
      CUR_YMD=${ANALYS_YMD}
      CUR_HOUR=${ANALYS_HOUR}
      n=$(( n + 1 ))
   done
}

# Stream calculated parameters into xargs to scale concurrently
generate_tasks | xargs -P 16 -n 5 bash -c 'process_analysis "$@"' _

exit 0

