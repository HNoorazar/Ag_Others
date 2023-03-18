#!/bin/bash
cd /home/h.noorazar/Sid/sidFabio/01_cumGDD_inTimeWindow_Oct18

outer=1
for veg_type in tomato carrot spinach strawberries
do
  for param_type in claudio
  do
    for model_type in observed # GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
    do
      for start_doy in 1 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345 360
      do
        cp q_countGDD_inTimeWindow_temp_nonLinear_Oct18.sh ./qsubs/q_countGDD_inTW_NL_Oct18_$outer.sh
        sed -i s/outer/"$outer"/g                          ./qsubs/q_countGDD_inTW_NL_Oct18_$outer.sh
        sed -i s/veg_type/"$veg_type"/g                    ./qsubs/q_countGDD_inTW_NL_Oct18_$outer.sh
        sed -i s/param_type/"$param_type"/g                ./qsubs/q_countGDD_inTW_NL_Oct18_$outer.sh
        sed -i s/model_type/"$model_type"/g                ./qsubs/q_countGDD_inTW_NL_Oct18_$outer.sh
        sed -i s/start_doy/"$start_doy"/g                  ./qsubs/q_countGDD_inTW_NL_Oct18_$outer.sh
        let "outer+=1" 
      done
    done
  done
done

