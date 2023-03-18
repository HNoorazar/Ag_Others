#
# d_cumGDD_nonLinear_Oct18_2022.R and associated files are 
# about the talk that Fabio has had with Claudio.
# They want to compute "right" maturity GDD based on 
# non-linear model and Claudios parameters within certain number of days!
## See oct 17th email from Fabio and google sheet to see the parameters.

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
veg_type   = args[1]
model_type = args[2]
param_type = args[3] # "fabio" or "claudio"

######################################################################
# Define main output path
database <- "/data/project/agaid/h.noorazar/sidFabio_FV/"
param_dir = file.path(paste0(database, "/000_parameters/")) # Kamiak

out_database = database
out_database_date = "_Oct18_2022"


file_list = "VIC_tomato_points_directory.csv"

local_files <- read.csv(paste0(param_dir, file_list))
local_files$full_file=paste0(local_files$path, model_type, "/rcp85/", local_files$file_name)
print (head(local_files, 2))

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params_Oct17_2022.csv"),  as.is=T))
veg_params=veg_params[veg_params$veg==veg_type]
print ("line 43")
print (veg_params)

# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()
#
#  on Kamiak everything has 8 variales
#
# future data are all over the place. West are in Adams directory
# non-west are elsewhere. Hence this if-else statement.
# right this second (Sept. 2022 we are doing observed and future (i.e. no modeled historical))

if (param_type=="fabio"){
  LC=veg_params$lower_cut
  UC=veg_params$upper_cut
  Topt=veg_params$ToptNonLinear
    }else{
  LC=veg_params$Claudio_lower_cut
  UC=veg_params$Claudio_upper_cut
  Topt=veg_params$Claudio_ToptNonLinear
  print (paste("line 60", LC, UC, Topt, sep=", "))
}

print (paste0("line 68, ", param_type, ", LC:  ", LC, ", UC: ", UC, ", Topt: ", Topt))
counter=1

if (model_type=="observed"){
  path_="/data/project/agaid/rajagopalan_agroecosystems/commondata/meteorologicaldata/gridded/gridMET/gridmet/historical/"
  for(file in local_files$file_name){
    met_data <- compute_GDD_nonLinear(data_dir = path_,
                                      file_name = file, 
                                      data_type_= model_type, 
                                      lower_cut = LC, 
                                      upper_cut = UC, 
                                      T_opt     = Topt)
    met_data <- met_data %>%
                select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
                data.table()

    current_out = paste0(out_database, "/00_cumGDD_separateLocationsModels/", veg_type, "/", 
                          gsub("-", "", model_type), "_nonLinear_", param_type, out_database_date, "/")

    if (dir.exists(current_out) == F) {dir.create(path = current_out, recursive = T)}
    write.csv(met_data, file = paste0(current_out, file, ".csv"), row.names=FALSE)

    }

} else{
      # path_= paste0("/data/adam/data/metdata/VIC_ext_maca_v2_binary_westUSA/", model_type, "/rcp85/")
      # for(file in local_files$full_file){
      #   print (paste0("line 86", file))
      #   if (file.exists(file)){
      #     print (paste0("line 88"))
      #     met_data <- compute_GDD_nonLinear(data_dir =path_,
      #                                       file_name =file, 
      #                                       data_type_=model_type, 
      #                                       lower_cut = LC, 
      #                                       upper_cut = UC, 
      #                                       T_opt     = Topt)
      #     met_data <- met_data %>%
      #                 select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
      #                 data.table()

      #     current_out = paste0(out_database, "/00_cumGDD_separateLocationsModels/", veg_type, "/", 
      #                           gsub("-", "", model_type), "_nonLinear_", param_type, out_database_date, "/")
      #     if (dir.exists(current_out) == F) {
      #         dir.create(path = current_out, recursive = T)}
      #     out_file_name = tail(stringr::str_split(string=file, pattern="/")[[1]], n=1) 
      #     write.csv(met_data, file = paste0(current_out, out_file_name, ".csv"),row.names=FALSE)
      #   }
      # }

      dir_base_1 = "/data/adam/data/metdata/VIC_ext_maca_v2_binary_westUSA/"
      dir_base_2 = "/data/project/agaid/rajagopalan_agroecosystems/commondata/meteorologicaldata/gridded/GCMs/US_Conus/"

      path_1 = paste0(dir_base_1, model_type, "/rcp85/")
      path_2 = paste0(dir_base_2, model_type, "/rcp85/")
      
      for(file in local_files$file_name){
        full_file_1 <- paste0(path_1, file)
        full_file_2 <- paste0(path_2, file)
        
        if (file.exists(full_file_1)){
          if (counter==1){print (paste0("line 129"))}

          met_data <- compute_GDD_nonLinear(data_dir  = path_1,
                                            file_name = full_file_1, 
                                            data_type_= model_type, 
                                            lower_cut = LC, 
                                            upper_cut = UC,
                                            T_opt     = Topt)
          met_data <- met_data %>%
                      select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
                      data.table()

          current_out = paste0(out_database, "/00_cumGDD_separateLocationsModels/", veg_type, "/", 
                               gsub("-", "", model_type), "_nonLinear_", param_type, out_database_date, "/")
           if (dir.exists(current_out) == F) {dir.create(path = current_out, recursive = T)}
          out_file_name = tail(stringr::str_split(string=file, pattern="/")[[1]], n=1) 
          write.csv(met_data, file = paste0(current_out, out_file_name, ".csv"),row.names=FALSE)
        }else if (file.exists(full_file_2)){

          met_data <- compute_GDD_nonLinear(data_dir  = path_2,
                                            file_name = full_file_2, 
                                            data_type_= model_type, 
                                            lower_cut = LC, 
                                            upper_cut = UC,
                                            T_opt     = Topt)

          met_data <- met_data %>%
                    select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
                    data.table()
          if (counter==1){
            print (paste0("line 158"))
            print (head(met_data, 2))
            counter=counter+1
          }

          current_out = paste0(out_database, "/00_cumGDD_separateLocationsModels/", veg_type, "/", 
                            gsub("-", "", model_type), "_nonLinear_", param_type, out_database_date, "/")
          if (dir.exists(current_out) == F) {
            dir.create(path = current_out, recursive = T)}
          out_file_name = tail(stringr::str_split(string=file, pattern="/")[[1]], n=1) 
          write.csv(met_data, file = paste0(current_out, out_file_name, ".csv"),row.names=FALSE)
         }
        }
}

print ("Line 105")
# print (current_out)
# How long did it take?
end_time <- Sys.time()
print(end_time - start_time)
