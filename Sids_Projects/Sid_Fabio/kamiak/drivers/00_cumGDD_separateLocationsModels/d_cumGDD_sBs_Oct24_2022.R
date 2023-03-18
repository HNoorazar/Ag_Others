
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

# .libPaths("/data/hydro/R_libs35")
# .libPaths()
library(data.table)

# source_path = "/Users/hn/Documents/00_GitHub/Ag/Sids_Projects/Sid_Fabio/kamiak/SidFabio_core.R"
source_path = "/home/h.noorazar/Sid/sidFabio/SidFabio_core.R"
source(source_path)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 9)
options(digit=9)

start_time <- Sys.time() # Time the processing of this batch of files
######################################################################
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################

args = commandArgs(trailingOnly=TRUE)
veg_type   = args[1]
#
# Later in the code we read "veg_params_Oct17_2022.csv" which 
# includes only Claudio's most recent parameters. So, hard coding here!
#
param_type = "claudio"    # args[2] # "fabio" or "claudio"
model_type = "observed"
######################################################################
# Define directories
database <- "/data/project/agaid/h.noorazar/sidFabio_FV/"

param_dir = paste0(database, "000_parameters/") # Kamiak

out_database = database
out_dir_postfix = "_Oct24/" # Check Oct. 24 email of Fabio
######################################################################
#####
##### Read parameters
#####
file_list_name = "VIC_noPasture_CRD_ID_unique.csv"

local_files <- data.table(read.csv(paste0(param_dir, file_list_name)))

# Pick up california
local_files <- local_files %>%
               filter(STASD_N %in% c(640, 650, 651, 680))%>%
               data.table()
#
# The following file includes only Claudio's most recent parameters.
#
claudio_veg_params <- data.table(read.csv(paste0(param_dir, "veg_params_Oct17_2022.csv"),  as.is=T))
claudio_veg_params=claudio_veg_params[claudio_veg_params$veg==veg_type]

linear_veg_params <- data.table(read.csv(paste0(param_dir, "veg_params.csv"),  as.is=T))
linear_veg_params=linear_veg_params[linear_veg_params$veg==veg_type]

print ("line 76")
print ("linear_veg_params: ")
print (linear_veg_params)
print ("____________________________________________________________________________________________________")
print ("claudio_veg_params: ")
print (claudio_veg_params)

# 3. Process the data -----------------------------------------------------

#
#  on Kamiak everything has 8 variales
#
# future data are all over the place. West are in Adams directory
# non-west are elsewhere. Hence this if-else statement.
# right this second (Sept. 2022 we are doing observed and future (i.e. no modeled historical))

if (model_type=="observed"){
  path_="/data/project/agaid/rajagopalan_agroecosystems/commondata/meteorologicaldata/gridded/gridMET/gridmet/historical/"
  for(file in local_files$file_name){
    if (file=="data_25.96875_-97.59375"){
      print (paste0("line 90: ", file))
      print (file.exists(paste0(path_, file)))}
    met_data <- compute_GDD_nonLinear(data_dir  = path_,
                                      file_name = file, 
                                      data_type_= model_type, 
                                      lower_cut = claudio_veg_params$Claudio_lower_cut, 
                                      upper_cut = claudio_veg_params$Claudio_upper_cut, 
                                      T_opt     = claudio_veg_params$Claudio_ToptNonLinear)
    met_data <- met_data %>%
                select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
                data.table()

    met_data_linear <- compute_GDD_linear(data_dir=path_,
                                          file_name=file, 
                                          data_type_=model_type, 
                                          lower_cut=linear_veg_params$lower_cut, 
                                          upper_cut=linear_veg_params$upper_cut)
    met_data_linear <- met_data_linear %>%
                       select(-c(precip, windspeed, SPH, Rmax, Rmin)) %>%
                       data.table()
    setnames(met_data_linear, old=c("daily_GDD"), new=c("linear_daily_GDD"))

    met_data <- cbind(met_data, met_data_linear[, "linear_daily_GDD"])

    current_out = paste0(out_database, "/00_cumGDD_separateLocationsModels/", veg_type, "/", 
                          gsub("-", "", model_type), "_sBs_", param_type, out_dir_postfix, "/")

    if (dir.exists(current_out) == F) {dir.create(path = current_out, recursive = T)}
    write.csv(met_data, file = paste0(current_out, file, ".csv"), row.names=FALSE)

    }
} else {
  print ("what's the matter with you")
}

print ("Line 122 - Last print")
# print (current_out)
# How long did it take?
end_time <- Sys.time()
print(end_time - start_time)
