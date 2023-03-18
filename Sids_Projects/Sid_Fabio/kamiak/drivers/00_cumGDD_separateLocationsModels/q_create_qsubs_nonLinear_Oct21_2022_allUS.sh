#!/bin/bash
cd /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels

outer=1
for veg_type in tomato carrot spinach strawberries 
do
  for param_type in claudio
  do
    for modelName in observed GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
    do
      cp q_cumGDD_template_nonLinear_Oct21_2022_allUS.sh ./qsubs/nonLinear_qcumGDD_Oct21_2022_allUS_$outer.sh
      sed -i s/outer/"$outer"/g                          ./qsubs/nonLinear_qcumGDD_Oct21_2022_allUS_$outer.sh
      sed -i s/veg_type/"$veg_type"/g                    ./qsubs/nonLinear_qcumGDD_Oct21_2022_allUS_$outer.sh
      sed -i s/modelName/"$modelName"/g                  ./qsubs/nonLinear_qcumGDD_Oct21_2022_allUS_$outer.sh
      sed -i s/param_type/"$param_type"/g                ./qsubs/nonLinear_qcumGDD_Oct21_2022_allUS_$outer.sh
      let "outer+=1" 
    done
  done
done


## d_cumGDD_nonLinear_Oct21_2022_allUS.R and associated files are 
## about the talk that Fabio has had with Claudio.
## He took average of medians of accumulated GDD over 104 days
## from planting dated where planting dates were march 15, .... etc.
## He does not believe there is a reason for Claudio to change
## this maturity GDDs.
##
