#!/bin/bash
cd /home/hnoorazar/Sid/sidFabio/02_aggregate_Maturiry_EE

outer=1
for veg_type in tomato
do
  ## for param_type in fabio claudio
    ## do
    cp q_aggregate_Maturiry_EE_temp_Linear.sh  ./qsubs/q_aggregateMaturity_EE_Linear$outer.sh
    sed -i s/outer/"$outer"/g                     ./qsubs/q_aggregateMaturity_EE_Linear$outer.sh
    sed -i s/veg_type/"$veg_type"/g               ./qsubs/q_aggregateMaturity_EE_Linear$outer.sh
    ## sed -i s/model_type/"$model_type"/g           ./qsubs/q_aggregateMaturity_EE_Linear$outer.sh
    ## sed -i s/start_doy/"$start_doy"/g             ./qsubs/q_aggregateMaturity_EE_Linear$outer.sh
    ## sed -i s/param_type/"$param_type"/g           ./qsubs/q_aggregateMaturity_EE_Linear$outer.sh
    let "outer+=1"
  ## done
done