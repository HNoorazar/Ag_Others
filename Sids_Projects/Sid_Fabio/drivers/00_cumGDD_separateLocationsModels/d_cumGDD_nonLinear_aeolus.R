.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)


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
veg_type = args[1]
model_type = args[2]
param_type=args[3] # "fabio" or "claudio"

######################################################################
# Define main output path
out_dir = "/data/hydro/users/Hossein/Sids_Projects/SidFabio/"

# 2a. Only use files in geographic locations we're interested in
param_dir = file.path("/data/hydro/users/Hossein/Sids_Projects/SidFabio/000_parameters/")

if (veg_type=="tomato"){
  file_list = "VIC_tomato_points.txt"
}

local_files <- read.delim(file = paste0(param_dir, file_list), 
                          header=FALSE, as.is=TRUE)
local_files <- as.vector(local_files$V1)

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params.csv"),  as.is=T))
veg_params=veg_params[veg_params$veg==veg_type]


# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()


if (model_type=="observed"){
  path_="/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"
  } else{
   path_= paste0("/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/", model_type, "/rcp85/")
}

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

for(file in local_files){
  met_data <- compute_GDD_nonLinear(data_dir=path_, 
                                    file_name=file, 
                                    observed_or_future=model_type, 
                                    lower_cut=LC, 
                                    upper_cut=UC, 
                                    T_opt=Topt)

  # 3b. Clean it up
  met_data <- met_data %>%
              select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
              data.table()


  current_out = paste0(out_dir, "/00_cumGDD_nonLinear/", veg_type, "/", gsub("-", "", model_type), "_", param_type, "/")
  if (dir.exists(current_out) == F) {
      dir.create(path = current_out, recursive = T)
    }

  write.csv(met_data, file = paste0(current_out, file, ".csv"), row.names=FALSE)
  
}

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)
