#!/bin/bash
cd /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels

outer=1
for veg_type in tomato
do
  for param_type in fabio claudio
  do
    for modelName in observed GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
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