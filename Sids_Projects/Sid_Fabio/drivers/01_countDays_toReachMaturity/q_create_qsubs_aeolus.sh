#!/bin/bash
cd /home/hnoorazar/Sid/sidFabio/01_countDays_toReachMaturity

outer=1
for veg_type in tomato
do
  for model_type in observed ## bcc-csm1-1 CNRM-CM5 HadGEM2-ES365 MIROC5 bcc-csm1-1-m CSIRO-Mk3-6-0 inmcm4 MIROC-ESM BNU-ESM GFDL-ESM2G IPSL-CM5A-LR MIROC-ESM-CHEM CanESM2 GFDL-ESM2M IPSL-CM5A-MR MRI-CGCM3 CCSM4 HadGEM2-CC365 IPSL-CM5B-LR NorESM1-M
  do
      for start_doy in 1 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345 360
      do
        cp q_countDaysMaturity_temp.sh  ./qsubs/q_countDaysMaturity$outer.sh
        sed -i s/outer/"$outer"/g           ./qsubs/q_countDaysMaturity$outer.sh
        sed -i s/veg_type/"$veg_type"/g     ./qsubs/q_countDaysMaturity$outer.sh
        sed -i s/model_type/"$model_type"/g ./qsubs/q_countDaysMaturity$outer.sh
        sed -i s/start_doy/"$start_doy"/g   ./qsubs/q_countDaysMaturity$outer.sh
        let "outer+=1" 
      done
  done
done