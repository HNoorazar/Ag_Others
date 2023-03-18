#!/bin/bash

cd /home/hnoorazar/Sid/sidFabio/01_countDays_toReachMaturity/qsubs

for runname in {1..180}
do
qsub ./q_countDaysMaturity_nonLinear$runname.sh
done
