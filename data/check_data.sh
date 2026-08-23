#!/bin/bash

# set -x

s3dir=s3://noaa-gfs-bdp-pds
totalyears=1
startyear=2024
res=0p25
#res=1p00
forecasthour=f006

regular_dir=terrain-regular-grid
icosahedral_dir=icosahedral-grid
datadir=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data

dayinmonth=(31 28 31 30 31 30 31 31 30 31 30 31)

keepgoing=true

flszlist=(46578519 46581849)

n=0
while [[ "${n}" -lt "${totalyears}" && "${keepgoing}" == "true" ]]
do
   YYYY=$(( startyear + n ))

   if (( (YYYY % 400 == 0) || (YYYY % 4 == 0 && YYYY % 100 != 0) )); then
      echo "$YYYY is a leap year."
      dayinmonth[1]=29
   else
      dayinmonth[1]=28
   fi

   month=1
   while [[ $month -le 12 && "${keepgoing}" == "true" ]]
   do
      if [ $month -lt 10 ]
      then
	 MM=0${month}
      else
	 MM=${month}
      fi

      nm=$(( month - 1 ))
      day=1
      while [[ $day -le ${dayinmonth[nm]} && "${keepgoing}" == "true" ]]
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
	   #iflnm=gfs.t${HH}z.pgrb2b.${res}.${forecasthour}
	    iflnm=gfs.t${HH}z.pgrb2.${res}.${forecasthour}
	    ncflnm=${regular_dir}/gfs.${YYYY}${MM}${DD}.t${HH}z.${res}.${forecasthour}.nc
	    mlgridflnm=${icosahedral_dir}/icosahedral_logstate_m6.${YYYY}${MM}${DD}.t${HH}z.${res}.${forecasthour}.nc

            if [ ! -f ${mlgridflnm} ]
            then
	       keepgoing=false
	      #tflnm=gribfiles/tmp_${YYYY}${MM}${DD}_gfs.t${HH}z.pgrb2b.${res}.${forecasthour}
	       tflnm=gribfiles/tmp_${YYYY}${MM}${DD}_gfs.t${HH}z.pgrb2.${res}.${forecasthour}
	       if [ ! -f ${ncflnm} ]
	       then
                  echo "aws s3 cp --no-sign-request ${s3dir}/${fdir}/${HH}/atmos/${iflnm} ${tflnm}"
	          echo "python interpolate_to_terrain_heights.py --etopo ${datadir}/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc --input ${tflnm} --output ${ncflnm}"
	       fi
	       echo "python interpolate_to_logstate_icosahedral.py -i ${ncflnm} -m ../graph/graph-grid/global_icosahedral_mesh_m6.nc -o ${mlgridflnm}"
	       break
            else
	       fs=$(stat -c %s ${mlgridflnm})
               if [[ ${fs} -ne ${flszlist[0]} && ${fs} -ne ${flszlist[1]} && ${fs} -ne ${flszlist[2]} && ${fs} -ne ${flszlist[3]} ]]
               then
	          keepgoing=false
		  echo "File ${mlgridflnm} is ${fs} did not match known file size ${flszlist[0]}, ${flszlist[1]} and ${flszlist[2]}"
	         #tflnm=tmp_${YYYY}${MM}${DD}_gfs.t${HH}z.pgrb2b.${res}.${forecasthour}
	          tflnm=tmp_${YYYY}${MM}${DD}_gfs.t${HH}z.pgrb2.${res}.${forecasthour}
                  echo "aws s3 cp --no-sign-request ${s3dir}/${fdir}/${HH}/atmos/${iflnm} ${tflnm}"
	          echo "python interpolate_to_terrain_heights.py --etopo ${datadir}/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc --input ${tflnm} --output ${ncflnm}"
	          echo "python interpolate_to_logstate_icosahedral.py -i ${ncflnm} -m ../graph/graph-grid/global_icosahedral_mesh_m6.nc -o ${mlgridflnm}"
	          break
               fi
            fi
         done
         day=$(( day + 1 ))
      done
      month=$(( month + 1 ))
   done

   n=$(( n + 1 ))
done

