#!/bin/bash
#SBATCH --partition=cahnrs,cahnrs_bigmem,cahnrs_gpu,kamiak,rajagopalan,stockle
#SBATCH --requeue
#SBATCH --job-name=cumGDD_outer # Job Name
#SBATCH --time=0-12:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=06GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000
###SBATCH -k o
#SBATCH --output=/home/h.noorazar/Sid/sidFabio/02_aggregate_Maturiry_EE/error/aggSpecial_outer.o
#SBATCH --error=/home/h.noorazar/Sid/sidFabio/02_aggregate_Maturiry_EE/error/aggSpecial_outer.e
echo
echo "--- We are now in $PWD, running an R script ..."
echo

echo
echo %A_%a
echo
echo $A_$a
echo "--- %A_%a --- "
echo
echo "---- $A_$a --- "
echo
module load r/4.1.0
Rscript --vanilla /home/h.noorazar/Sid/sidFabio/02_aggregate_Maturiry_EE/d_aggregate_Maturiry_EE_Linear_special_order.R veg_type


echo
echo "----- DONE -----"
echo

exit 0

