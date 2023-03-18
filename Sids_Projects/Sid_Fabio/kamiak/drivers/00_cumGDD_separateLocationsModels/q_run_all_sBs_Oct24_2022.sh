#!/bin/bash

cd /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/qsubs

for runname in {1..25}
do
sbatch ./sBs_cumGDD_Oct24_2022_$runname.sh
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