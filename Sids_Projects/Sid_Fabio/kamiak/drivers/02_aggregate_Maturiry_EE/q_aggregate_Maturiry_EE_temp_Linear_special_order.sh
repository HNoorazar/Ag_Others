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

## echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."

# Load R on compute node
module load r/4.1.0
## cd /data/project/agaid/AnalogData_Sid/Creating_Variables_old/
### Rscript --vanilla ./sid_script_west_model_pr_ch.R ${SLURM_ARRAY_TASK_ID}

Rscript --vanilla /home/h.noorazar/Sid/sidFabio/02_aggregate_Maturiry_EE/d_aggregate_Maturiry_EE_Linear_special_order.R veg_type


echo
echo "----- DONE -----"
echo

exit 0

