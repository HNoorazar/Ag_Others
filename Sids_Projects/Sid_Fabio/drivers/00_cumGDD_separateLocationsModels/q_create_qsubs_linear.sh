#!/bin/bash
cd /home/hnoorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels

outer=1
for veg_type in tomato
do
  for modelName in observed GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M bcc-csm1-1 CNRM-CM5 MIROC5 bcc-csm1-1-m CSIRO-Mk3-6-0 inmcm4 MIROC-ESM BNU-ESM GFDL-ESM2G CanESM2 IPSL-CM5A-MR MRI-CGCM3 CCSM4 HadGEM2-CC365 IPSL-CM5B-LR
  do
    cp q_cumGDD_template_linear.sh    ./qsubs/q_cumGDD_$outer.sh
    sed -i s/outer/"$outer"/g         ./qsubs/q_cumGDD_$outer.sh
    sed -i s/veg_type/"$veg_type"/g   ./qsubs/q_cumGDD_$outer.sh
    sed -i s/modelName/"$modelName"/g ./qsubs/q_cumGDD_$outer.sh
    let "outer+=1" 
  done
done