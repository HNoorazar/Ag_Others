#!/bin/bash
#SBATCH --partition=cahnrs,cahnrs_bigmem,cahnrs_gpu,kamiak,rajagopalan,stockle
#SBATCH --requeue
#SBATCH --job-name=00_NLOct18_outer # Job Name
#SBATCH --time=2-24:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=06GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/error/nonLinear_cumGDD_outer_Oc18.o
#SBATCH --error=/home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/error/nonLinear_cumGDD_outer_Oc18.e

echo
echo "--- We are now in $PWD, running an R script ..."
echo

# Load R on compute node
module load r/4.1.0
cd /data/project/agaid/AnalogData_Sid/Creating_Variables_old/
### Rscript --vanilla ./sid_script_west_model_pr_ch.R ${SLURM_ARRAY_TASK_ID}


Rscript --vanilla /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/d_cumGDD_nonLinear_Oct18_2022.R veg_type modelName param_type

echo
echo "----- DONE -----"
echo

exit 0

## d_cumGDD_nonLinear_Oct18_2022.R and associated files are 
## about the talk that Fabio has had with Claudio.
## They want to compute "right" maturity GDD based on 
## non-linear model and Claudios parameters within certain number of days!
## See oct 17th email from Fabio and google sheet to see the parameters.
