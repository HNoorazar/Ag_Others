#!/bin/bash
#SBATCH --partition=cahnrs,cahnrs_bigmem,cahnrs_gpu,kamiak,rajagopalan,stockle
#SBATCH --requeue
#SBATCH --job-name=linNonLin_Agg_outer # Job Name
#SBATCH --time=0-12:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=06GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/h.noorazar/Sid/sidFabio/02_aggregate_Maturiry_EE/error/agg_lin_nonLin_outer.o
#SBATCH  --error=/home/h.noorazar/Sid/sidFabio/02_aggregate_Maturiry_EE/error/agg_lin_nonLin_outer.e
echo
echo "--- We are now in $PWD, running an R script ..."
echo

# Load R on compute node
module load r/4.1.0
Rscript --vanilla /home/h.noorazar/Sid/sidFabio/02_aggregate_Maturiry_EE/d_aggregate_Maturiry_EE_linear_nonLinear.R veg_type

echo
echo "----- DONE -----"
echo

exit 0

