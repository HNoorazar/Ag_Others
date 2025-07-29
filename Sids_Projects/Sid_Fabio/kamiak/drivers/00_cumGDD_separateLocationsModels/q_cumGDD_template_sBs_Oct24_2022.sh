#!/bin/bash
#SBATCH --partition=cahnrs,cahnrs_bigmem,cahnrs_gpu,kamiak,rajagopalan,stockle
#SBATCH --requeue
#SBATCH --job-name=00_NLOct24_outer # Job Name
#SBATCH --time=2-24:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=06GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/error/sBs_cumGDD_outer_Oc24.o
#SBATCH --error=/home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/error/sBs_cumGDD_outer_Oc24.e

echo
echo "--- We are now in $PWD, running an R script ..."
echo

# Load R on compute node
module load r/4.1.0
cd /data/project/agaid/AnalogData_Sid/Creating_Variables_old/
### Rscript --vanilla ./sid_script_west_model_pr_ch.R ${SLURM_ARRAY_TASK_ID}


Rscript --vanilla /home/h.noorazar/Sid/sidFabio/00_cumGDD_separateLocationsModels/d_cumGDD_sBs_Oct24_2022.R veg_type

echo
echo "----- DONE -----"
echo

exit 0



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