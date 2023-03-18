#!/bin/bash

cd /home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity/qsubs

for runname in {1..300}
do
sbatch ./q_countDaysMaturity_nonLinear$runname.sh
done
