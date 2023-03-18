#!/bin/bash
cd /home/hnoorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels

outer=1
for veg_type in tomato
do
  for param_type in fabio claudio
  do
    for modelName in observed # bcc-csm1-1 CNRM-CM5 HadGEM2-ES365 MIROC5 bcc-csm1-1-m CSIRO-Mk3-6-0 inmcm4 MIROC-ESM BNU-ESM GFDL-ESM2G IPSL-CM5A-LR MIROC-ESM-CHEM CanESM2 GFDL-ESM2M IPSL-CM5A-MR MRI-CGCM3 CCSM4 HadGEM2-CC365 IPSL-CM5B-LR NorESM1-M
    do
      cp q_cumGDD_template_nonLinear.sh   ./qsubs/nonLinear_qcumGDD_$outer.sh
      sed -i s/outer/"$outer"/g           ./qsubs/nonLinear_qcumGDD_$outer.sh
      sed -i s/veg_type/"$veg_type"/g     ./qsubs/nonLinear_qcumGDD_$outer.sh
      sed -i s/modelName/"$modelName"/g   ./qsubs/nonLinear_qcumGDD_$outer.sh
      sed -i s/param_type/"$param_type"/g ./qsubs/nonLinear_qcumGDD_$outer.sh
      let "outer+=1" 
    done
  done
done