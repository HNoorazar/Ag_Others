#!/bin/bash

#PBS -V
#PBS -N row_number

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=40gb
#PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/supriya/error/row_number.e
#PBS -o /home/hnoorazar/supriya/error/row_number.o
#PBS -m abe

cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs/merge/
echo
echo We are in the $PWD directory
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla convert_supriya_SFs.R row_number

echo
echo "----- DONE -----"
echo

exit 0


