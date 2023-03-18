.libPaths("/data/hydro/R_libs35")
.libPaths()

rm(list=ls())

library(dplyr)
library(data.table)
library(stringr)

source_path = "/home/hnoorazar/Sid/sidFabio/SidFabio_core.R"
source(source_path)
options(digit=9)
options(digits=9)

######################################################################
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################

args = commandArgs(trailingOnly=TRUE)
veg_type = args[1] # carrot, tomato, spinach, strawberry
# model_type = args[2] # observed or name of future models; e.g. BNU-XYZ
# start_doy = strtoi(args[3]) # 1, 15, 30, 45, ...
## param_type=args[2] # "fabio" or "claudio"
######################################################################
dir_base <- "/data/hydro/users/Hossein/Sids_Projects/SidFabio/01_days2maturity_EE_linear/"
out_dir_base = "/data/hydro/users/Hossein/Sids_Projects/SidFabio/"

# 2a. Only use files in geographic locations we're interested in
param_dir = file.path("/data/hydro/users/Hossein/Sids_Projects/SidFabio/000_parameters/")

# if (veg_type=="tomato"){
#   file_list = "VIC_tomato_points.txt"
# }

# local_files <- read.delim(file = paste0(param_dir, file_list), 
#                           header=FALSE, as.is=TRUE)
# local_files <- as.vector(local_files$V1)

# veg_params <- data.table(read.csv(paste0(param_dir, "veg_params.csv"),  as.is=T))
# veg_params=veg_params[veg_params$veg==veg_type, ]

tomato_crd_trial=data.table(read.csv(paste0(param_dir, "tomato_crd_trial.csv"),  as.is=T))

# -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()

curr_in <- paste0(dir_base, veg_type, "/") # param_type
csv_file_list <- list.files(path=curr_in, pattern = "csv")

annual_means_within_CRD = data.table()
median_of_annual_means_within_CRD_within_TP = data.table()

for (a_file in csv_file_list){
  curr_file <- data.table(read.csv(paste0(curr_in, a_file), as.is=TRUE))
  break_file=stringr::str_split(string=a_file, pattern="_")[[1]]
  curr_model = break_file[1]
  curr_startDoY = break_file[4]
  
  curr_file <- dplyr::left_join(x=curr_file, y=tomato_crd_trial, by = "location")
  curr_file <- data.table(curr_file)
  
  curr_annual_means = curr_file[, .(mean_days_to_maturity = mean(days_to_maturity), 
                                    mean_no_days_in_opt_interval = mean(no_days_in_opt_interval), 
                                    mean_no_of_extreme_cold = mean(no_of_extreme_cold), 
                                    mean_no_of_extreme_heat = mean(no_of_extreme_heat)),
                                    by = c("STASD_N", "year")]

  curr_annual_means$model = curr_model
  curr_annual_means$startDoY = curr_startDoY
  curr_annual_means$veg = veg_type
  # curr_annual_means$param_type = param_type

  curr_annual_means <- data.table(curr_annual_means)

  curr_medians_TP = curr_annual_means[, .(median_of_mean_days_to_maturity = median(mean_days_to_maturity), 
                                          median_of_mean_no_days_in_opt_interval = median(mean_no_days_in_opt_interval), 
                                          median_of_mean_no_of_extreme_cold = median(mean_no_of_extreme_cold), 
                                          median_of_mean_no_of_extreme_heat = median(mean_no_of_extreme_heat)),
                                          by = c("STASD_N")]
  curr_medians_TP$model = curr_model
  curr_medians_TP$startDoY = curr_startDoY
  curr_medians_TP$veg = veg_type
  # curr_medians_TP$param_type = param_type

  # concat the result to the big dataframe
  annual_means_within_CRD=rbind(annual_means_within_CRD, curr_annual_means)
  median_of_annual_means_within_CRD_within_TP=rbind(median_of_annual_means_within_CRD_within_TP, curr_medians_TP)

}


current_out = paste0(out_dir_base, "/02_aggregate_Maturiry_EE_linear/", veg_type, "/") # "_", model_type, 
if (dir.exists(current_out) == F) {
    dir.create(path = current_out, recursive = T)
}

write.csv(annual_means_within_CRD, 
          file = paste0(current_out, "annual_means_within_CRD.csv"), 
          row.names=FALSE)

write.csv(median_of_annual_means_within_CRD_within_TP, 
          file = paste0(current_out, "within_TP_median_of_annual_means_within_CRD.csv"), 
          row.names=FALSE)

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)



