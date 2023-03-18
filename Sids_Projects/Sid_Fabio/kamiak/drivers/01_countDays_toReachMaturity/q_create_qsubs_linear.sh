#!/bin/bash
cd /home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity

outer=1
for veg_type in tomato
do
  for model_type in observed GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
  do
    for start_doy in 1 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345 360
    do
      cp q_countDays2Maturity_temp_linear.sh ./qsubs/q_countDaysMaturity_linear$outer.sh
      sed -i s/outer/"$outer"/g              ./qsubs/q_countDaysMaturity_linear$outer.sh
      sed -i s/veg_type/"$veg_type"/g        ./qsubs/q_countDaysMaturity_linear$outer.sh
      sed -i s/model_type/"$model_type"/g    ./qsubs/q_countDaysMaturity_linear$outer.sh
      sed -i s/start_doy/"$start_doy"/g      ./qsubs/q_countDaysMaturity_linear$outer.sh
      let "outer+=1" 
    done
  done
done

