#!/bin/bash

cd /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/qsubs

for runname in {1..23}
do
sbatch ./q_cumGDD_$runname.sh
done
