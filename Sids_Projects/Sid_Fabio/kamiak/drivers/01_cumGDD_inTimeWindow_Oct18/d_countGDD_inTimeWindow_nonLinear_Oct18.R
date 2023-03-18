#
# Oct. 18. 2022
#
# .libPaths("/data/hydro/R_libs35")
# .libPaths()
library(data.table)
source_path = "/home/h.noorazar/Sid/sidFabio/SidFabio_core.R"
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
model_type = args[2] # observed or name of future models; e.g. BNU-XYZ
start_doy = strtoi(args[3]) # 1, 15, 30, 45, ...
param_type=args[4] # "fabio" or "claudio"

model_type=gsub("-", "", model_type)
######################################################################
database  <- "/data/project/agaid/h.noorazar/sidFabio_FV/"
param_dir <- paste0(database, "/000_parameters/")

in_database <- paste0(database, "00_cumGDD_separateLocationsModels/")
data_dir <- paste0(in_database, veg_type, "/", model_type, "_nonLinear_", param_type, "_Oct18_2022/")

out_database <- database # kamiak
param_dir <- file.path("/home/h.noorazar/Sid/sidFabio/000_parameters/") # Kamiak

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params_Oct17_2022.csv"),  as.is=T))
veg_params <- veg_params[veg_params$veg==veg_type, ]
print (veg_params)

VIC_grids <- data.table(read.csv(paste0(param_dir, "tomato_crd_trial_680_2_CA80.csv")))
VIC_grids <- VIC_grids[VIC_grids$CRD %in% c("CA40", "CA50", "CA51", "CA80", "FL50", "MI80")]

grid_count <- dim(VIC_grids)[1]

local_files <- VIC_grids$location

# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()

col_names <- c("location", "year", "cumGDD_inTW", "cumSRAD_inTW", "no_of_extreme_cold", "no_of_extreme_heat")

# 36 below comes from 1980-2015.
#     on Oct. 18 they wanted to do 1980-2010
if (model_type=="observed"){
  start_yr=1980
  end_yr=2010
 } else{
  start_yr=2041
  end_yr=2070 
}

year_count = end_yr-start_yr+1
days_to_maturity_tb <- setNames(data.table(matrix(nrow = grid_count*year_count, 
                                                  ncol = length(col_names))), col_names)
days_to_maturity_tb$location <- rep.int(VIC_grids$location, year_count)
setorderv(days_to_maturity_tb,  ("location"))
days_to_maturity_tb$year <- rep.int(c(start_yr:end_yr), grid_count)


setnames(VIC_grids, old=c("location"), new=c("file_name"))
# strsplit vector 
x <- sapply(VIC_grids$file_name, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
lat = x[2, ]
long = x[3, ]
VIC_grids$lat=lat
VIC_grids$long=long
VIC_grids$location= paste0(VIC_grids$lat, "_" , VIC_grids$long)

for(fileName in VIC_grids$file_name){

  # Look into the right directory
  if (file.exists(paste0(data_dir, fileName, ".csv"))){
    data_tb <- data.table(read.csv(paste0(data_dir, fileName, ".csv")))

    if (model_type=="observed"){
      data_tb <- data_tb[data_tb$year>=1980]
      data_tb <- data_tb[data_tb$year<=2010]
    }
    data_tb$row_num <- seq.int(nrow(data_tb))

    # add day of year
    data_tb$doy = 1
    data_tb[, doy := cumsum(doy), by=list(year)]
    #
    # Here is where we have filtered 2041 to 2070: unique(days_to_maturity_tb$year)
    #
    for (a_year in sort(unique(days_to_maturity_tb$year))){
      curr_row_num <- data_tb[data_tb$year==a_year & data_tb$doy==start_doy, ]$row_num
      curr_data <- data_tb[data_tb$row_num>=curr_row_num & data_tb$row_num<=(curr_row_num+veg_params$baseline_cycle-1),]

      curr_data <- data.table(curr_data)
      curr_data$cum_GDD <- 0
      curr_data[, cum_GDD := cumsum(WEDD)] # , by=list(year)

      curr_data$cum_SRAD <- 0
      curr_data[, cum_SRAD := cumsum(SRAD)] # , by=list(year)

      days_to_maturity_tb$cumGDD_inTW[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year] = tail(curr_data, 1)$cum_GDD
      days_to_maturity_tb$cumSRAD_inTW[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year] = tail(curr_data, 1)$cum_SRAD

      extreme_cold_tb = curr_data %>%
                        filter(tmin <= veg_params$cold_stress) %>%
                        data.table()
      days_to_maturity_tb$no_of_extreme_cold[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year] = dim(extreme_cold_tb)[1]

      extreme_heat_tb = curr_data %>%
                        filter(tmax>=veg_params$heat_stress) %>%
                        data.table()

      days_to_maturity_tb$no_of_extreme_heat[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year] = dim(extreme_heat_tb)[1]

    }
  }
}

current_out = paste0(out_database, "/01_cumGDD_inTimeWindow_Oct18/", 
                     veg_type, "_nonlinear_", param_type, "/") 

if (dir.exists(current_out) == F) {
    dir.create(path = current_out, recursive = T)
}

write.csv(days_to_maturity_tb, 
          file = paste0(current_out, model_type, "_start_DoY_", start_doy, "_cumGDD_inTW_EE.csv"), 
          row.names=FALSE)

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)
