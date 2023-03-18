#!/bin/bash
cd /home/h.noorazar/Sid/sidFabio/02_aggregate_Maturiry_EE

outer=1

for veg_type in tomato
do
  cp q_aggregate_Maturiry_EE_temp_linear_nonLinear.sh  ./qsubs/q_aggregateMaturity_EE_linear_nonLinear$outer.sh
  sed -i s/outer/"$outer"/g                            ./qsubs/q_aggregateMaturity_EE_linear_nonLinear$outer.sh
  sed -i s/veg_type/"$veg_type"/g                      ./qsubs/q_aggregateMaturity_EE_linear_nonLinear$outer.sh
  ## sed -i s/model_type/"$model_type"/g               ./qsubs/q_aggregateMaturity_EE_linear_nonLinear$outer.sh
  ## sed -i s/start_doy/"$start_doy"/g                 ./qsubs/q_aggregateMaturity_EE_linear_nonLinear$outer.sh
  ## sed -i s/param_type/"$param_type"/g               ./qsubs/q_aggregateMaturity_EE_linear_nonLinear$outer.sh
  let "outer+=1"
done