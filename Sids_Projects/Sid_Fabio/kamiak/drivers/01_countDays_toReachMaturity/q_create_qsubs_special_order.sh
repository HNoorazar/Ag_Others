#!/bin/bash
cd /home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity

outer=1
for veg_type in tomato
do
  for model_type in observed GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
  do
    for start_doy in 104 87 83 6 134 67 70 67 5 119 47 60 55 3 114
    do
      cp q_countDaysMaturity_temp_special_order.sh  ./qsubs/q_countDaysMaturity_special_order$outer.sh
      sed -i s/outer/"$outer"/g                     ./qsubs/q_countDaysMaturity_special_order$outer.sh
      sed -i s/veg_type/"$veg_type"/g               ./qsubs/q_countDaysMaturity_special_order$outer.sh
      sed -i s/model_type/"$model_type"/g           ./qsubs/q_countDaysMaturity_special_order$outer.sh
      sed -i s/start_doy/"$start_doy"/g             ./qsubs/q_countDaysMaturity_special_order$outer.sh
      let "outer+=1" 
    done
  done
done