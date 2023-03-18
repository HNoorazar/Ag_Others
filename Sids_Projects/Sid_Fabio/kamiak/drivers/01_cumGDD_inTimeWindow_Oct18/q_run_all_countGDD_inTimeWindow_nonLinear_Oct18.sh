#!/bin/bash

cd /home/h.noorazar/Sid/sidFabio/01_cumGDD_inTimeWindow_Oct18/qsubs

for runname in {1..300}
do
sbatch ./q_countGDD_inTW_NL_Oct18_$runname.sh
done
