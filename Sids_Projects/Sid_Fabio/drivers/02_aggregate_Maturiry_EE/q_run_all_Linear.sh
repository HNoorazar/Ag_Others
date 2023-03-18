#!/bin/bash

cd /home/hnoorazar/Sid/sidFabio/02_aggregate_Maturiry_EE/qsubs

for runname in {1..180}
do
qsub ./q_aggregateMaturity_EE_Linear$runname.sh
done
