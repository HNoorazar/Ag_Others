#!/bin/bash

cd /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/qsubs

for runname in {1..25}
do
sbatch ./nonLinear_qcumGDD_$runname.sh
done
