# .libPaths("/data/hydro/R_libs35")
# .libPaths()
library(data.table)


source_path = "/home/h.noorazar/Sid/sidFabio/SidFabio_core.R"
source(source_path)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 9)
options(digit=9)

######################################################################
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################

args = commandArgs(trailingOnly=TRUE)
veg_type = args[1]
model_type = args[2]

######################################################################
# Define main output path
# 2a. Only use files in geographic locations we're interested in
database <- "/data/project/agaid/h.noorazar/sidFabio_FV/"
param_dir = file.path(paste0(database, "/000_parameters/")) # Kamiak
out_database = database

if (veg_type=="tomato"){
  file_list = "VIC_tomato_points_directory.csv"
}

# local_files <- read.delim(file = paste0(param_dir, file_list), 
#                           header=FALSE, as.is=TRUE)
# local_files <- as.vector(local_files$V1)

local_files <- read.csv(paste0(param_dir, file_list))
local_files$full_file=paste0(local_files$path, model_type, "/rcp85/", local_files$file_name)
print (head(local_files, 2))

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params.csv"),  as.is=T))
veg_params=veg_params[veg_params$veg==veg_type]


# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()


#
#  on Kamiak everything has 8 variales
#
# future data are all over the place. West are in Adams directory
# non-west are elsewhere. Hence this if-else statement.
# right this second (Sept. 2022 we are doing observed and future (i.e. no modeled historical))


if (model_type=="observed"){
  path_="/data/project/agaid/rajagopalan_agroecosystems/commondata/meteorologicaldata/gridded/gridMET/gridmet/historical/"

  for(file in local_files$file_name){
    met_data <- compute_GDD_linear(data_dir=path_,
                                   file_name=file, 
                                   data_type_=model_type, 
                                   lower_cut=veg_params$lower_cut, 
                                   upper_cut=veg_params$upper_cut)
    met_data <- met_data %>%
                select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
                data.table()
    current_out = paste0(out_database, "/00_cumGDD_separateLocationsModels/", veg_type, "/", gsub("-", "", model_type), "/")
    if (dir.exists(current_out) == F) {
        dir.create(path = current_out, recursive = T)
      }
    write.csv(met_data, file = paste0(current_out, file, ".csv"),row.names=FALSE)

    }
  } else {
   path_= paste0("/data/adam/data/metdata/VIC_ext_maca_v2_binary_westUSA/", model_type, "/rcp85/")
   for(file in local_files$full_file){
      if (file.exists(file)){
                              met_data <- compute_GDD_linear(data_dir=path_,
                                                             file_name=file, 
                                                             data_type_=model_type, 
                                                             lower_cut=veg_params$lower_cut, 
                                                             upper_cut=veg_params$upper_cut)
                              met_data <- met_data %>%
                                          select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
                                          data.table()

                               current_out = paste0(out_database, "/00_cumGDD_separateLocationsModels/", veg_type, "/", gsub("-", "", model_type), "/")
                              if (dir.exists(current_out) == F) {
                                dir.create(path = current_out, recursive = T)}
      
                              out_file_name = tail(stringr::str_split(string=file, pattern="/")[[1]], n=1) 
                              write.csv(met_data, file = paste0(current_out, out_file_name, ".csv"),row.names=FALSE)
    }
   }

}

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)
