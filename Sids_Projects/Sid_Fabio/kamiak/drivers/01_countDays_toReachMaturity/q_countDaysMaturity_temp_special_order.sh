#!/bin/bash
#SBATCH --partition=cahnrs,cahnrs_bigmem,cahnrs_gpu,kamiak,rajagopalan,stockle
#SBATCH --requeue
#SBATCH --job-name=01_SO_outer # Job Name
#SBATCH --time=0-12:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=06GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity/error/matureEE_special_order_outer.o
#SBATCH --error=/home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity/error/matureEE_special_order_outer.e
echo
echo "--- We are now in $PWD, running an R script ..."
echo

# Load R on compute node
module load r/4.1.0
Rscript --vanilla /home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity/d_countDays_to_maturity_special_order.R veg_type model_type start_doy 

echo
echo "----- DONE -----"
echo

exit 0
