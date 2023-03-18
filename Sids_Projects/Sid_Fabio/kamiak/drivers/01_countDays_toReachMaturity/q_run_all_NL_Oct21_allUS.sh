#!/bin/bash

cd /home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity/qsubs

for runname in {1..600}
do
sbatch ./q_countDaysMaturity_NL_Oct21_allUS_$runname.sh
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
