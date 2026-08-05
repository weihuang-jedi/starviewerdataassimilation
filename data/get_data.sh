#!/bin/bash
#SBATCH --job-name=s3_data_transfer
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=epic
#SBATCH --output=log.s3_transfer_%j.out

set -x

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

# Define the processing logic as a concurrent Bash function
process_hour() {
   local YYYY=$1
   local MM=$2
   local DD=$3
   local HH=$4
   local s3dir=$5
   local fdir=$6
   local res=$7
   local fcst=$8

   local iflnm=gfs.t${HH}z.pgrb2b.${res}.${fcst}
   local tflnm=tmp_${YYYY}${MM}${DD}.gfs.t${HH}z.pgrb2.${res}.${fcst}
   local ncflnm=regular_grid/gfs.${YYYY}${MM}${DD}.t${HH}z.${res}.${fcst}.nc
   local mlgridflnm=icosahedral_grid/global_icosahedral_m4.${YYYY}${MM}${DD}.t${HH}z.${res}.${fcst}.nc

   if [ ! -f "${mlgridflnm}" ]; then
      if [ ! -f "${ncflnm}" ]; then
         if [ ! -f "${tflnm}" ]; then
            aws s3 cp --no-sign-request "${s3dir}/${fdir}/${HH}/atmos/${iflnm}" "${tflnm}"
         fi

         # Note: Fixed argument mismatch. Passed $tflnm instead of non-existent $iflnm
         python interpolate_to_heights.py -i "${tflnm}" -o "${ncflnm}"
         rm -f "${tflnm}" *.idx
      fi

      # Run the icosahedral interpolation
      python interpolate_to_logstate_icosahedral.py \
         --input "${ncflnm}" \
         --mesh ./graph/global_icosahedral_mesh_m4.nc \
         --output "${mlgridflnm}"
   fi
}

# Configuration
s3dir=s3://noaa-gfs-bdp-pds
totalyears=1
startyear=2024
res=1p00
fcst=f006

dayinmonth=(31 28 31 30 31 30 31 31 30 31 30 31)

n=0
while [ "${n}" -lt "${totalyears}" ]
do
   YYYY=$(( startyear + n ))

   # Leap year calculation
   if (( (YYYY % 400 == 0) || (YYYY % 4 == 0 && YYYY % 100 != 0) )); then
      echo "$YYYY is a leap year."
      dayinmonth[1]=29
   else
      dayinmonth[1]=28
   fi

   month=1
   while [ $month -le 12 ]
   do
      if [ $month -lt 10 ]; then
         MM=0${month}
      else
         MM=${month}
      fi

      nm=$(( month - 1 ))
      day=1
      while [ $day -le ${dayinmonth[nm]} ]
      do
         if [ $day -lt 10 ]; then
            DD=0${day}
         else
            DD=${day}
         fi

         fdir=gfs.${YYYY}${MM}${DD}
         
         # Fire all 4 hour functions asynchronously in the background
         for HH in 00 06 12 18
         do
            process_hour "${YYYY}" "${MM}" "${DD}" "${HH}" "${s3dir}" "${fdir}" "${res}" "${fcst}" &
         done
         
         # CRITICAL: Wait for all 4 background processes of this day to finish
         wait

         day=$(( day + 1 ))
      done
      month=$(( month + 1 ))
   done

   n=$(( n + 1 ))
done

exit 0

