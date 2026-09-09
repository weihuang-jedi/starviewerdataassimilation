#!/bin/bash
#SBATCH --job-name=get_data
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=epic
#SBATCH --output=log.get_data_%j.out

set -x

## cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerweathermodel/data
# cd /scratch5/purged/Wei.Huang/src/starviewerweathermodel/data
cd /scratch5/purged/Wei.Huang/src/starviewerdataassimilation/data

totalyears=1
startyear=2026
#export res=1p00
export res=0p25
export fcst=f000

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

# Define the processing logic as a standard Bash function
process_hour() {
   local YYYY=$1
   local MM=$2
   local DD=$3
   local HH=$4
   local s3dir=$5
   local fdir=$6
   local res=$7
   local fcst=$8

   # local iflnm=gfs.t${HH}z.pgrb2b.${res}.${fcst}
   local iflnm=gfs.t${HH}z.pgrb2.${res}.${fcst}
   local tflnm=gribfiles/tmp_${YYYY}${MM}${DD}.gfs.t${HH}z.pgrb2.${res}.${fcst}
   local ncflnm=terrain-regular-grid/gfs.${YYYY}${MM}${DD}.t${HH}z.${res}.${fcst}.nc
   local mlgridflnm=icosahedral-truth/icosahedral_logstate_m6.${YYYY}${MM}${DD}.t${HH}z.${res}.${fcst}.nc
   local datadir=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data

   if [ ! -f "${mlgridflnm}" ]; then
      if [ ! -f "${ncflnm}" ]; then
         if [ ! -f "${tflnm}" ]; then
            aws s3 cp --no-sign-request "${s3dir}/${fdir}/${HH}/atmos/${iflnm}" "${tflnm}"
         fi

	 python interpolate_to_terrain_heights.py \
            --input "${tflnm}" \
            --etopo ${datadir}/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc \
            --output "${ncflnm}"

	 if [ ! -f ${ncflnm} ]; then
	    echo "Error to generate: ${ncflnm}. stop"
	    exit 1
	 fi
         rm -f "${tflnm}" "${tflnm}*.idx"
      fi

      # Run the icosahedral interpolation
      python interpolate_to_logstate_icosahedral.py \
         -i "${ncflnm}" \
         -m ../graph/graph-grid/global_icosahedral_mesh_m6.nc \
         -o "${mlgridflnm}"
      if [ ! -f ${mlgridflnm} ]; then
         echo "Error to generate: ${mlgridflnm}. stop"
         exit 1
      fi
   fi
}

# Export the function and variables so subshells spawned by xargs can read them
export -f process_hour
export s3dir=s3://noaa-gfs-bdp-pds
dayinmonth=(31 28 31 30 31 30 31 31 30 31 30 31)

# Function to generate the stream of tasks
generate_tasks() {
   local n=0
   while [ "${n}" -lt "${totalyears}" ]
   do
      local YYYY=$(( startyear + n ))

      if (( (YYYY % 400 == 0) || (YYYY % 4 == 0 && YYYY % 100 != 0) )); then
         dayinmonth[1]=29
      else
         dayinmonth[1]=28
      fi

      local month=1
      # local month=5
      # local month=9
      while [ $month -le 12 ]
      do
         printf -v MM "%02d" $month
         local nm=$(( month - 1 ))
         
         local day=1
         while [ $day -le ${dayinmonth[nm]} ]
         do
            printf -v DD "%02d" $day
            local fdir=gfs.${YYYY}${MM}${DD}

            for HH in 00 06 12 18
            do
               # Print arguments separated by space, one task per line
               echo "${YYYY} ${MM} ${DD} ${HH} ${s3dir} ${fdir} ${res} ${fcst}"
            done
            day=$(( day + 1 ))
         done
         month=$(( month + 1 ))
      done
      n=$(( n + 1 ))
   done
}

# Stream tasks into xargs to run exactly 16 concurrently
generate_tasks | xargs -P 4 -n 8 bash -c 'process_hour "$@"' _

exit 0

