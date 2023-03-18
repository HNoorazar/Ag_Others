##################################
##
##  Sept. 22. Table for Result section of overleaf file. 
##
rm(list=ls())
library(data.table)
library(dplyr)
library(stringr)

library(dplyr)
library(data.table)
library(ggplot2)

source_path = "/home/hnoorazar/Sid/sidFabio/SidFabio_core_plot.R"
source_path = "/Users/hn/Documents/00_GitHub/Ag/Sids_Projects/Sid_Fabio/SidFabio_core_plot.R"
source(source_path)
options(digits=9)

################################
########
######## parameters
########
param_dir = "/Users/hn/Documents/01_research_data/Sid/SidFabio/parameters/"
tomato_crd_trial = read.csv(paste0(param_dir, "tomato_crd_trial.csv"))
tomato_crd_trial <- within(tomato_crd_trial, remove(location))
tomato_crd_trial<-data.table(tomato_crd_trial)

veg_type = "tomato"  # "tomato" at this point, later: "carrot", "spinach", "strawberry", "tomato"

################################
########
######## Directories
########

out_dir_base = "/Users/hn/Documents/01_research_data/Sid/SidFabio/03_table_results/"
data_dir_base = "/Users/hn/Documents/01_research_data/Sid/SidFabio/02_aggregate_Maturiry_EE_Kamiak/"
sub_dirs = c("tomato_linear", "tomato_nonlinear_claudio", "tomato_nonlinear_fabio")
file_names = c("annual_means_within_CRD.csv", "within_TP_median_of_annual_means_within_CRD.csv")

for (a_dir in sub_dirs){

  data_dir = paste0(data_dir_base, a_dir, "/")
  out_dir = paste0(out_dir_base, a_dir, "/")
  if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}
  ######## read
  annual_means_within_CRD = data.table(read.csv(paste0(data_dir, file_names[1])))
  within_TP_median_of_annual_means_within_CRD = data.table(read.csv(paste0(data_dir, file_names[2])))
  ################################
  ########
  ########    Subset
  ########

  unique(annual_means_within_CRD$STASD_N)

  # 12 is Florida and 26 is Michigan.
  chosen_CRD <- c(640, 650, 651, 1250, 2680)
  chosen_CRD_alph <- c("CA40", "CA50", "CA51", "FL50", "MI80")
  
  annual_means_within_CRD <- annual_means_within_CRD %>%
                             filter(STASD_N %in% chosen_CRD) %>%
                             data.table()
  within_TP_median_of_annual_means_within_CRD <- within_TP_median_of_annual_means_within_CRD %>%
                                                 filter(STASD_N %in% chosen_CRD) %>%
                                                 data.table()

  alphabet_CRD <- data.table()
  alphabet_CRD$STASD_N <- chosen_CRD
  alphabet_CRD$CRD <- chosen_CRD_alph
  annual_means_within_CRD <- dplyr::left_join(x=annual_means_within_CRD, y=alphabet_CRD, by="STASD_N")

  within_TP_median_of_annual_means_within_CRD <- dplyr::left_join(x=within_TP_median_of_annual_means_within_CRD, 
                                                                  y=alphabet_CRD, by="STASD_N")

  annual_means_within_CRD$CRD <- factor(annual_means_within_CRD$CRD, 
                                        levels=chosen_CRD_alph, order=TRUE)

  within_TP_median_of_annual_means_within_CRD$CRD <- factor(within_TP_median_of_annual_means_within_CRD$CRD, 
                                                            levels = chosen_CRD_alph, order=TRUE)

  start_DOY = sort(unique(annual_means_within_CRD$startDoY))
  annual_means_within_CRD$startDoY <- factor(annual_means_within_CRD$startDoY, 
                                             levels = start_DOY, order=TRUE)
  within_TP_median_of_annual_means_within_CRD$startDoY <- factor(within_TP_median_of_annual_means_within_CRD$startDoY, 
                                                                 levels=start_DOY, order=TRUE)

  median_of_means_4col <- within_TP_median_of_annual_means_within_CRD[, c("CRD", "time_period", "startDoY",
                                                                           "median_of_mean_days_to_maturity")]

  table_to_export = reshape(median_of_means_4col, idvar=c("startDoY", "time_period"), timevar="CRD", direction="wide")
  table_to_export <- table_to_export[order(startDoY, ),]

  start_col_to_round=3
  cols_to_round <- names(table_to_export)[start_col_to_round:length(names(table_to_export))]

  # round the numbers
  table_to_export[,(cols_to_round) := round(.SD, 0), .SDcols=cols_to_round]

  # change the column names. They are long, but descriptive:
  x <- sapply(colnames(table_to_export)[start_col_to_round: length(colnames(table_to_export))], 
                function(x) strsplit(x, "\\.")[[1]], 
                USE.NAMES=FALSE)
  new_colNames = x[2, ]

  setnames(table_to_export, 
           old=colnames(table_to_export)[start_col_to_round: length(colnames(table_to_export))], 
           new=new_colNames)


  # change the order of columns
  table_to_export = table_to_export[, c(1, 2, 5, 6, 4, 3, 7)]
  table_to_export=table_to_export[order(-rank(time_period), startDoY)]

  write.csv(table_to_export, 
            file = paste0(out_dir, "median_of_means_of_days_to_maturity_wide.csv"), 
            row.names=FALSE)

  ######### SR 

  median_of_means_4col <- within_TP_median_of_annual_means_within_CRD[, c("CRD", "time_period", 
                                                                          "median_of_mean_of_cum_solar", "startDoY")]

  table_to_export = reshape(median_of_means_4col, idvar=c("startDoY", "time_period"), timevar="CRD", direction="wide")
  table_to_export <- table_to_export[order(startDoY, ),]

  cols_to_round <- names(table_to_export)[start_col_to_round:length(names(table_to_export))]

  # round the numbers
  table_to_export[,(cols_to_round) := round(.SD,0), .SDcols=cols_to_round]

  # change the column names. They are long, but descriptive:
   x <- sapply(colnames(table_to_export)[start_col_to_round: length(colnames(table_to_export))], 
               function(x) strsplit(x, "\\.")[[1]], 
               USE.NAMES=FALSE)
  new_colNames = x[2, ]

  setnames(table_to_export, 
           old=colnames(table_to_export)[start_col_to_round: length(colnames(table_to_export))], 
           new=new_colNames)


  # change the order of columns
  table_to_export = table_to_export[, c(1, 2, 5, 6, 4, 3, 7)]
  table_to_export=table_to_export[order(-rank(time_period), startDoY)]
  write.csv(table_to_export, 
            file = paste0(out_dir, "median_of_means_of_cumSR_wide.csv"), 
            row.names=FALSE)
}








