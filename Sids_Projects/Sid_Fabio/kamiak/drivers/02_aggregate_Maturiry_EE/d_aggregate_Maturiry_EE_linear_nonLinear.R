# .libPaths("/data/hydro/R_libs35")
# .libPaths()

rm(list=ls())

library(dplyr)
library(data.table)
library(stringr)

source_path = "/home/h.noorazar/Sid/sidFabio/SidFabio_core.R"
source(source_path)
options(digit=9)
options(digits=9)

# -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()


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

param_dir = file.path("/home/h.noorazar/Sid/sidFabio/000_parameters/") # Kamiak
dir_base <- "/home/h.noorazar/Sid/sidFabio/01_countDays_toReachMaturity/"
out_dir_base = "/home/h.noorazar/Sid/sidFabio/" # kamiak

if (veg_type=="tomato"){
  tomato_crd_trial=data.table(read.csv(paste0(param_dir, "tomato_crd_trial.csv"),  as.is=T))
}


# list of folders/directories where each directory contains
# data from a given start_DoY
#
subdirs_ <- c(paste0(veg_type, "_linear"), 
             paste0(veg_type, "_nonlinear_claudio"), 
             paste0(veg_type, "_nonlinear_fabio"))

for (a_subdir in subdirs_){
  data_dir = paste0(dir_base, "/", a_subdir, "/")

  annual_means_within_CRD = data.table()
  median_of_annual_means_within_CRD_within_TP = data.table()

  #####################################################################################
    
  # for a given start_DoY do:

  # List of CSV files in the data directory:
  csv_file_list <- list.files(path=data_dir , pattern = "csv")
  all_data <- data.table()
  print (csv_file_list)
  for (a_file in csv_file_list){
    curr_file <- data.table(read.csv(paste0(data_dir, "/", a_file), as.is=TRUE))
    break_file=stringr::str_split(string=a_file, pattern="_")[[1]]
    curr_model = break_file[1]
    curr_startDoY = break_file[4]
    print (a_file)
    print (curr_model)
    print (curr_startDoY)

    curr_file$model = curr_model
    curr_file$startDoY = curr_startDoY
    curr_file$veg = veg_type

    if (curr_model=="observed"){
      curr_file$time_period = "observed"

      }else{
      curr_file$time_period = "2050s"
    }
    print (unique(curr_file$time_period))
    curr_file <- dplyr::left_join(x=curr_file, y=tomato_crd_trial, by = "location")
    curr_file <- data.table(curr_file)
    all_data <- rbind(all_data, curr_file)
  }

  given_DoY_annual_means_within_CRD=all_data[, .(mean_days_to_maturity = mean(days_to_maturity), 
                                                 mean_no_days_in_opt_interval = mean(no_days_in_opt_interval), 
                                                 mean_no_of_extreme_cold = mean(no_of_extreme_cold),
                                                 mean_no_of_extreme_heat = mean(no_of_extreme_heat),
                                                 mean_of_cum_solar = mean(cum_solar)),
                                                 by = c("STASD_N", "year", "startDoY", "time_period")]

  given_DoY_median_of_annual_means_within_CRD_within_TP=given_DoY_annual_means_within_CRD[,.(median_of_mean_days_to_maturity=median(mean_days_to_maturity), 
                                                                                             median_of_mean_no_days_in_opt_interval=median(mean_no_days_in_opt_interval), 
                                                                                             median_of_mean_no_of_extreme_cold=median(mean_no_of_extreme_cold), 
                                                                                             median_of_mean_no_of_extreme_heat=median(mean_no_of_extreme_heat),
                                                                                             median_of_mean_of_cum_solar=median(mean_of_cum_solar)),
                                                                                             by = c("STASD_N", "startDoY", "time_period")]

  annual_means_within_CRD = rbind(annual_means_within_CRD, given_DoY_annual_means_within_CRD)
  median_of_annual_means_within_CRD_within_TP = rbind(median_of_annual_means_within_CRD_within_TP, 
                                                      given_DoY_median_of_annual_means_within_CRD_within_TP)

  #####################################################################################

  current_out = paste0(out_dir_base, "/02_aggregate_Maturiry_EE/", a_subdir, "/")
  if (dir.exists(current_out) == F) {
    dir.create(path = current_out, recursive = T)
  }

  write.csv(annual_means_within_CRD, 
            file = paste0(current_out, "annual_means_within_CRD.csv"), 
            row.names=FALSE)

  write.csv(median_of_annual_means_within_CRD_within_TP, 
            file = paste0(current_out, "within_TP_median_of_annual_means_within_CRD.csv"), 
            row.names=FALSE)

}

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)

