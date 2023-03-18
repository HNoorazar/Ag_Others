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

# args = commandArgs(trailingOnly=TRUE)
# veg_type = args[1] # carrot, tomato, spinach, strawberry
# model_type = args[2] # observed or name of future models; e.g. BNU-XYZ
# start_doy = strtoi(args[3]) # 1, 15, 30, 45, ...
## param_type=args[2] # "fabio" or "claudio"
######################################################################

database <- "/data/project/agaid/h.noorazar/sidFabio_FV/"
param_dir = file.path(paste0(database, "/000_parameters/")) # Kamiak

in_database <- paste0(database, "/01_cumGDD_inTimeWindow_Oct18/")
out_database = database

tomato_crd_trial=data.table(read.csv(paste0(param_dir, "tomato_crd_trial_680_2_CA80.csv"),  as.is=T))

# list of folders/directories where each directory contains
# data from a given start_DoY
#
veggies <- c("carrot", "spinach", "strawberries", "tomato")

all_means <- data.table()
all_medians <- data.table()

for (veg_type in veggies){

  data_dir = paste0(in_database, veg_type, "_nonlinear_claudio/")

  # annual_means_within_CRD = data.table()
  # median_of_annual_means_within_CRD_within_TP = data.table()

  #####################################################################################
    
  # for a given start_DoY do:
  # List of CSV files in the data directory:
  csv_file_list <- list.files(path=data_dir , pattern = "csv")
  all_data_of_veg_type <- data.table()
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
    curr_file$veg_type = veg_type

    if (curr_model=="observed"){
      curr_file$time_period = "observed"

      }else{
      curr_file$time_period = "2050s"
    }
    print (unique(curr_file$time_period))
    curr_file <- dplyr::left_join(x=curr_file, y=tomato_crd_trial, by = "location")
    curr_file <- data.table(curr_file)
    
    curr_file$state[curr_file$CRD %in% c("CA80", "CA51", "CA40", "CA50")]="CA"
    curr_file$state[curr_file$CRD %in% c("MI80")]="MI"
    curr_file$state[curr_file$CRD %in% c("FL50")]="FL"

    all_data_of_veg_type <- rbind(all_data_of_veg_type, curr_file)
  }


  means_data_a_veg_type=all_data_of_veg_type[, .(mean_cumGDD_inTW        = mean(cumGDD_inTW), 
                                                 mean_cumSRAD_inTW       = mean(cumSRAD_inTW), 
                                                 mean_no_of_extreme_cold = mean(no_of_extreme_cold),
                                                 mean_no_of_extreme_heat = mean(no_of_extreme_heat)),
                                                 by = c("startDoY", "veg_type", "time_period", "state", "year")]


  medians_of_means_data_a_veg_type=means_data_a_veg_type[, .(median_of_means_cumGDD_inTW        = median(mean_cumGDD_inTW), 
                                                             median_of_means_cumSRAD_inTW       = median(mean_cumSRAD_inTW), 
                                                             median_of_means_no_of_extreme_cold = median(mean_no_of_extreme_cold),
                                                             median_of_means_no_of_extreme_heat = median(mean_no_of_extreme_heat)),
                                                             by = c("startDoY", "veg_type", "time_period", "state")]

  

  all_medians = rbind(all_medians, medians_of_means_data_a_veg_type)
  #####################################################################################
}

current_out = paste0(out_database, "/02_aggregate_cumGDD_inTimeWindow_Oct18/")
if (dir.exists(current_out) == F) {
  dir.create(path = current_out, recursive = T)
}

states=c("CA", "MI", "FL")
all_medians$state    <- factor(all_medians$state,    levels=states, order=TRUE)
all_medians$startDoY <- factor(all_medians$startDoY, levels=sort(unique(all_medians$startDoY)), order=TRUE)
all_medians$veg_type <- factor(all_medians$veg_type, levels=sort(unique(all_medians$veg_type)), order=TRUE)

setorder(all_medians, cols = "state", "veg_type", "startDoY")
write.csv(all_medians, 
          file = paste0(current_out, "cumGDDMedians_of_means_withinState_inTW_allVegs_allstartDoYs.csv"), 
          row.names=FALSE)

CA_medians = all_medians[all_medians$state=="CA", ]

CA_medians$state    <- factor(CA_medians$state,    levels=states, order=TRUE)
CA_medians$startDoY <- factor(CA_medians$startDoY, levels=sort(unique(CA_medians$startDoY)), order=TRUE)
CA_medians$veg_type <- factor(CA_medians$veg_type, levels=sort(unique(CA_medians$veg_type)), order=TRUE)


setorder(CA_medians, cols = "state", "veg_type", "startDoY")

write.csv(CA_medians, 
          file = paste0(current_out, "CA_cumGDDMedians_of_means_withinState_inTW_allVegs_allstartDoYs.csv"), 
          row.names=FALSE)

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)

