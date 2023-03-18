#!/bin/bash

cd /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/qsubs

for runname in {1..24}
do
sbatch ./nonLinear_qcumGDD_Oct21_2022_allUS_$runname.sh
done


## d_cumGDD_nonLinear_Oct21_2022_allUS.R and associated files are 
## about the talk that Fabio has had with Claudio.
## He took average of medians of accumulated GDD over 104 days
## from planting dated where planting dates were march 15, .... etc.
## He does not believe there is a reason for Claudio to change
## this maturity GDDs.
##