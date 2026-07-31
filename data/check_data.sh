#!/bin/bash
s3dir=s3://noaa-gfs-bdp-pds

totalyears=1
startyear=2024
#res=0p25
res=1p00
forecasthour=f000

if [ "${forecasthour}" == "f000" ]; then
   regular_dir=regular_truth
   icosahedral_dir=icosahedral_truth
else
   regular_dir=regular_grid
   icosahedral_dir=icosahedral_grid
fi

dayinmonth=(31 28 31 30 31 30 31 31 30 31 30 31)

keepgoing=true

flszlist=(4225193 4228002 4230068 4233136)

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
	    iflnm=gfs.t${HH}z.pgrb2b.${res}.${forecasthour}
	    ncflnm=${regular_dir}/gfs.${YYYY}${MM}${DD}.t${HH}z.${res}.${forecasthour}.nc
	    mlgridflnm=${icosahedral_dir}/global_icosahedral_m4.${YYYY}${MM}${DD}.t${HH}z.${res}.${forecasthour}.nc

            if [ ! -f ${mlgridflnm} ]
            then
	       keepgoing=false
	       tflnm=tmp_${YYYY}${MM}${DD}_gfs.t${HH}z.pgrb2b.${res}.${forecasthour}
	       if [ ! -f ${ncflnm} ]
	       then
                  echo "aws s3 cp --no-sign-request ${s3dir}/${fdir}/${HH}/atmos/${iflnm} ${tflnm}"
	          echo "python interpolate_to_heights.py -i ${tflnm} -o ${ncflnm}"
	       fi
	       echo "python interpolate2icosahedral.py --input ${ncflnm} --mesh graph/global_icosahedral_mesh_m4.nc --output ${mlgridflnm}"
	       break
            else
	       fs=$(stat -c %s ${mlgridflnm})
               if [[ ${fs} -ne ${flszlist[0]} && ${fs} -ne ${flszlist[1]} && ${fs} -ne ${flszlist[2]} && ${fs} -ne ${flszlist[3]} ]]
               then
	          keepgoing=false
		  echo "File ${mlgridflnm} is ${fs} did not match known file size ${flszlist[0]}, ${flszlist[1]} and ${flszlist[2]}"
	          tflnm=tmp_${YYYY}${MM}${DD}_gfs.t${HH}z.pgrb2b.${res}.${forecasthour}
                  echo "aws s3 cp --no-sign-request ${s3dir}/${fdir}/${HH}/atmos/${iflnm} ${tflnm}"
	          echo "python interpolate_to_heights.py -i ${tflnm} -o ${ncflnm}"
	          echo "python interpolate2icosahedral.py --input ${ncflnm} --mesh graph/global_icosahedral_mesh_m4.nc --output ${mlgridflnm}"
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

