#!/bin/bash

cd /home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity/qsubs

for runname in {1..180}
do
sbatch ./q_countDaysMaturity_special_order$runname.sh
done
