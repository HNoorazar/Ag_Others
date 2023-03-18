#!/bin/bash

cd /home/hnoorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/qsubs

for runname in {1..2}
do
qsub ./nonLinear_qcumGDD_$runname.sh
done
