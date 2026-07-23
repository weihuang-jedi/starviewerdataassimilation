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

# aws s3 sync --no-sign-request s3://noaa-gfs-bdp-pds/gfs.20210202 gfs.20210202
# s3://noaa-gfs-bdp-pds/gfs.20210202
s3dir=s3://noaa-gfs-bdp-pds

totalyears=1
startyear=2023
#res=0p25
res=1p00
fcst=f006

dayinmonth=(31 28 31 30 31 30 31 31 30 31 30 31)

n=0
while [ "${n}" -lt "${totalyears}" ]
do
   YYYY=$(( startyear + n ))

   if (( (YYYY % 400 == 0) || (YYYY % 4 == 0 && YYYY % 100 != 0) )); then
      echo "$YYYY is a leap year."
      dayinmonth[1]=29
   else
      dayinmonth[1]=28
   fi

   month=1
   while [ $month -le 12 ]
   do
      if [ $month -lt 10 ]
      then
	 MM=0${month}
      else
	 MM=${month}
      fi

      nm=$(( month - 1 ))
      day=1
      while [ $day -le ${dayinmonth[nm]} ]
      do
         if [ $day -lt 10 ]
         then
	    DD=0${day}
         else
	    DD=${day}
         fi

         fdir=gfs.${YYYY}${MM}${DD}
         for HH in 00 06 12 18
         do
	    iflnm=gfs.t${HH}z.pgrb2b.${res}.${fcst}
	   #iflnm=gfs.t${HH}z.pgrb2.${res}.${fcst}
	    ncflnm=regular_grid/gfs.${YYYY}${MM}${DD}.t${HH}z.${res}.${fcst}.nc
	    mlgridflnm=icosahedral_grid/global_icosahedral_m4.${YYYY}${MM}${DD}.t${HH}z.${res}.${fcst}.nc

            if [ ! -f ${mlgridflnm} ]
            then
               if [ ! -f ${ncflnm} ]
               then
                  if [ ! -f ${iflnm} ]
                  then
                     aws s3 cp --no-sign-request ${s3dir}/${fdir}/${HH}/atmos/${iflnm} ${iflnm}
                    #aws s3 cp --no-sign-request ${s3dir}/${fdir}/${HH}/${iflnm} ${iflnm}
                  fi

	          python interpolate_to_heights.py -i ${iflnm} -o ${ncflnm}
	          rm -f ${iflnm} *.idx
               fi

	       python interpolate2icosahedral.py \
                  --input ${ncflnm} \
                  --mesh ./graph/global_icosahedral_mesh_m4.nc \
                  --output ${mlgridflnm} &
            fi
         done
	 wait
         day=$(( day + 1 ))
      done
      month=$(( month + 1 ))
   done

   n=$(( n + 1 ))
done

