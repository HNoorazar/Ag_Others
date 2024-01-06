#!/bin/bash

cd /home/h.noorazar/educational/qsubs

for arg1 in 1 2 3 4 5
do
  sbatch ./parallel_job_$arg1.sh
done