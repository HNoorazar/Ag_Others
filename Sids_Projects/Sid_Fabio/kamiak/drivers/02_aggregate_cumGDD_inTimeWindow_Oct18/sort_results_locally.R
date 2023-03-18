library(dplyr)
library(data.table)
library(stringr)


dir_base <- "/Users/hn/Documents/01_research_data/Sid/SidFabio/02_aggregate_cumGDD_inTimeWindow_Oct18/"

csv_file_list <- list.files(path=dir_base , pattern = "csv")

states=c("CA", "MI", "FL")
for (a_file in csv_file_list){
  curr_file <- data.table(read.csv(paste0(dir_base, "/", a_file), as.is=TRUE))
  curr_file$state    <- factor(curr_file$state,    levels=states, order=TRUE)
  curr_file$startDoY <- factor(curr_file$startDoY, levels=sort(unique(curr_file$startDoY)), order=TRUE)
  curr_file$veg_type <- factor(curr_file$veg_type, levels=sort(unique(curr_file$veg_type)), order=TRUE)
  setorder(curr_file, cols = "state", "veg_type", "startDoY")
  
  write.csv(curr_file, 
          file = paste0(dir_base, a_file), 
          row.names=FALSE)
  #####################################################################################
}








