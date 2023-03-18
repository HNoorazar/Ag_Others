#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N outer_veg_type_model_type_start_doy

## Define compute options
#PBS -l nodes=1:ppn=1
#PBS -l mem=2gb
#PBS -l walltime=06:00:00
##PBS -q hydro

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /home/hnoorazar/Sid/sidFabio/01_countDays_toReachMaturity/error/mature_EE_outer_E
#PBS -o /home/hnoorazar/Sid/sidFabio/01_countDays_toReachMaturity/error/mature_EE_outer_O

## Define path for reporting
#PBS -m abe

echo
echo We are in the $PWD directory
echo

echo
echo We are now in $PWD.
echo

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0
module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
module load r/3.5.1

Rscript --vanilla /home/hnoorazar/Sid/sidFabio/01_countDays_toReachMaturity/d_countDays_to_maturity.R veg_type model_type start_doy

echo
echo "----- DONE -----"
echo

exit 0
