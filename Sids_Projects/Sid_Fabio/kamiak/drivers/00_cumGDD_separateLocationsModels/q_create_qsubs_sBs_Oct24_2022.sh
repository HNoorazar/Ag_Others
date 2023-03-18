#!/bin/bash
cd /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels

outer=1
for veg_type in tomato carrot spinach strawberries
do
  for param_type in claudio
  do
    for modelName in observed # GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
    do
      cp q_cumGDD_template_sBs_Oct24_2022.sh ./qsubs/sBs_cumGDD_Oct24_2022_$outer.sh
      sed -i s/outer/"$outer"/g              ./qsubs/sBs_cumGDD_Oct24_2022_$outer.sh
      sed -i s/veg_type/"$veg_type"/g        ./qsubs/sBs_cumGDD_Oct24_2022_$outer.sh
      sed -i s/modelName/"$modelName"/g      ./qsubs/sBs_cumGDD_Oct24_2022_$outer.sh
      sed -i s/param_type/"$param_type"/g    ./qsubs/sBs_cumGDD_Oct24_2022_$outer.sh
      let "outer+=1" 
    done
  done
done




##
## Oct 25. Even tho Fabio did not think Claudio will 
## change the plan, Claudio changed it and wants to see NL_cumGDD and L_cumGDD side by side.
##  
## Probably based on something, they believe nonlinear is the better model.
## but after seeing the accumulated GDD they think it is not right!
##

##
## d_cumGDD_nonLinear_Oct21_2022_allUS.R and associated files are 
## about the talk that Fabio has had with Claudio.
## He took average of medians of accumulated GDD over 104 days
## from planting dated where planting dates were march 15, .... etc.
## Fabio does not believe there is a reason for Claudio to change
## this maturity GDDs.
##