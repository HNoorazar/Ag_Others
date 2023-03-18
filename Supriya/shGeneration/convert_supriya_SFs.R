rm(list=ls())
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(chillR)
library(tidyverse)
library(lubridate)

##############################################################################################################
#
#    Directory setup
#
data_dir <- paste0("/data/suriya/")
param_dir <- paste0("/data/suriya/parameters/")


########################################
#
# Terminal Argument. This is where in .sh file you pass an argument
# to this code.
#
args = commandArgs(trailingOnly=TRUE)
row_number = as.numeric(args[1])

########################################

file_names_table <- data.table(read.csv(file=paste0(param_dir, "file_names.csv"), 
                                        header=TRUE, as.is=TRUE))

# I have forgotten how you subset a row in R! 
current_file_name <- file_names_table[row_number, "file_name"]

current_file <- read_whatever it is. CSV? Raster? text?

# the rest of the code. Apply whatever to current_fime


