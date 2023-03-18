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
veg_type = args[1] # carrot, tomato, spinach, strawberry
model_type = args[2] # observed or name of future models; e.g. BNU-XYZ
start_doy = strtoi(args[3]) # 1, 15, 30, 45, ...

######################################################################
# Define main output path
dir_base <- "/data/hydro/users/Hossein/Sids_Projects/SidFabio/00_cumGDD_linear/"
data_dir <- paste0(dir_base, veg_type, "/", gsub("-", "", model_type), "/")
out_dir_base = "/data/hydro/users/Hossein/Sids_Projects/SidFabio/"

# 2a. Only use files in geographic locations we're interested in
param_dir = file.path("/data/hydro/users/Hossein/Sids_Projects/SidFabio/000_parameters/")

if (veg_type=="tomato"){
  file_list = "VIC_tomato_points.txt"
}

local_files <- read.delim(file = paste0(param_dir, file_list), 
                          header=FALSE, as.is=TRUE)
local_files <- as.vector(local_files$V1)

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params.csv"),  as.is=T))
veg_params=veg_params[veg_params$veg==veg_type, ]


# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()

col_names <- c("location", "year", "days_to_maturity", 
               "no_days_in_opt_interval", "no_of_extreme_cold", "no_of_extreme_heat")

# 36 below comes from 1980-2015.
if (model_type=="observed"){
  days_to_maturity_tb <- setNames(data.table(matrix(nrow = length(local_files)*36, 
                                                 ncol = length(col_names))), col_names)

  days_to_maturity_tb$location <- rep.int(local_files, 36)
  setorderv(days_to_maturity_tb,  ("location"))
  days_to_maturity_tb$year <- rep.int(c(1980:2015), length(local_files))
}


# x <- sapply(local_files, 
#             function(x) strsplit(x, "_")[[1]], 
#             USE.NAMES=FALSE)
# lat = x[2, ]
# long = x[3, ]

for(fileName in local_files){
  # The following function will look into the right directory when 
  data_tb <- data.table(read.csv(paste0(data_dir, fileName, ".csv")))
  data_tb$row_num <- seq.int(nrow(data_tb))

  # data_tb <- data_tb %>%
  #         select(-c(precip, windspeed, SPH, SRAD, Rmax, Rmin))

  # data_tb[, cum_GDD:=NULL] # cum_GDD is useless. I need to do this on yearly basis.

  for (a_year in sort(unique(days_to_maturity_tb$year))){
    curr_row_num <- data_tb[data_tb$year==a_year & data_tb$doy==start_doy, ]$row_num
    curr_data <- data_tb[data_tb$row_num>=curr_row_num, ]
    curr_data <- data.table(curr_data)
    curr_data$cum_GDD <- 0
    curr_data[, cum_GDD := cumsum(daily_GDD)] # , by=list(year)

    day_of_maturity=curr_data[curr_data$cum_GDD >= veg_params[veg_params$veg==veg_type]$maturity_gdd]
    dayCount = day_of_maturity$row_num[1]-curr_data$row_num[1]

    # Record days_to_maturity
    days_to_maturity_tb$days_to_maturity[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year] = dayCount

    # Subset the table between start day and day of maturity to count
    # the number of days in optimum interval and extreme events
    start_to_maturity_tb = curr_data %>%
                           filter(row_num>=curr_data$row_num[1]) %>%
                           filter(row_num<=day_of_maturity$row_num[1]) %>%
                           data.table()
    
    # optimum_table = start_to_maturity_tb %>%
    #                 filter(tmin>=veg_params$optimum_low) %>%
    #                 filter(tmax<=veg_params$optimum_hi) %>%
    #                 data.table()

    optimum_table = start_to_maturity_tb %>%
                    filter(Tavg>=veg_params$optimum_low) %>%
                    filter(Tavg<=veg_params$optimum_hi) %>%
                    data.table()

    days_to_maturity_tb$no_days_in_opt_interval[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year] = dim(optimum_table)[1]

    extreme_cold_tb = start_to_maturity_tb %>%
                      filter(tmin<=veg_params$cold_stress) %>%
                      data.table()

    days_to_maturity_tb$no_of_extreme_cold[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year] = dim(extreme_cold_tb)[1]


    extreme_heat_tb = start_to_maturity_tb %>%
                      filter(tmax>=veg_params$heat_stress) %>%
                      data.table()

    days_to_maturity_tb$no_of_extreme_heat[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year] = dim(extreme_heat_tb)[1]

  }  
}

current_out = paste0(out_dir_base, "/01_days2maturity_EE_linear/", veg_type, "/") # "_", model_type, 
if (dir.exists(current_out) == F) {
    dir.create(path = current_out, recursive = T)
}

write.csv(days_to_maturity_tb, 
          file = paste0(current_out, gsub("-", "", model_type), "_start_DoY_", start_doy, "_days2maturity_EE.csv"), 
          row.names=FALSE)



# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)
