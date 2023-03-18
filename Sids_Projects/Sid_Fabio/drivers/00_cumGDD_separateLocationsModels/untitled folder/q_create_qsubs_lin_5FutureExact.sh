#!/bin/bash
cd /home/hnoorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels

outer=1
for veg_type in tomato
do
  for modelName in observed GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
  do
    cp q_cumGDD_template.sh           ./qsubs/q_cumGDD_$outer.sh
    sed -i s/outer/"$outer"/g         ./qsubs/q_cumGDD_$outer.sh
    sed -i s/veg_type/"$veg_type"/g   ./qsubs/q_cumGDD_$outer.sh
    sed -i s/modelName/"$modelName"/g ./qsubs/q_cumGDD_$outer.sh
    let "outer+=1" 
  done
done