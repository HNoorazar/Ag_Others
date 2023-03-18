#!/bin/bash

cd /home/hnoorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/qsubs

for runname in {1..180}
do
qsub ./q_cumGDD_$runname.sh
done
