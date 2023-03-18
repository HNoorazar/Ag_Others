#!/bin/bash
cd /home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity

outer=1
for veg_type in tomato spinach strawberries carrot
do
  for param_type in claudio
  do
    for model_type in observed GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
    do
      for start_doy in 1 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345 360
      do
        cp q_countDays2Maturity_temp_NL_Oct21_allUS.sh ./qsubs/q_countDaysMaturity_NL_Oct21_allUS_$outer.sh
        sed -i s/outer/"$outer"/g                      ./qsubs/q_countDaysMaturity_NL_Oct21_allUS_$outer.sh
        sed -i s/veg_type/"$veg_type"/g                ./qsubs/q_countDaysMaturity_NL_Oct21_allUS_$outer.sh
        sed -i s/param_type/"$param_type"/g            ./qsubs/q_countDaysMaturity_NL_Oct21_allUS_$outer.sh
        sed -i s/model_type/"$model_type"/g            ./qsubs/q_countDaysMaturity_NL_Oct21_allUS_$outer.sh
        sed -i s/start_doy/"$start_doy"/g              ./qsubs/q_countDaysMaturity_NL_Oct21_allUS_$outer.sh
        let "outer+=1"
      done
    done
  done
done

##
## Oct. 21. 2022
##
## d_countDays_to_maturity_NL_Oct21_allUS.R and associated files are 
## about the talk that Fabio has had with Claudio.
## He took average of medians of accumulated GDD over 104 days
## from planting dated where planting dates were march 15, .... etc.
## He does not believe there is a reason for Claudio to change
## this maturity GDDs.
##
