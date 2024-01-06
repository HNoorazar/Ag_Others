#!/bin/bash
cd /home/h.noorazar/educational

outer=1
for arg1 in 1 2 3 4 5
do
  cp template.sh           ./qsubs/parallel_job_$arg1.sh
  sed -i s/arg1/"$arg1"/g  ./qsubs/parallel_job_$arg1.sh
  let "outer+=1"
done