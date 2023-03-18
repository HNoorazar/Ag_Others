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
# Time the processing of this batch of files
start_time <- Sys.time()

args = commandArgs(trailingOnly=TRUE)
veg_type = args[1] # carrot, tomato, spinach, strawberry
model_type = args[2] # observed or name of future models; e.g. BNU-XYZ
start_doy = args[3]

######################################################################
# Define directories
database <- "/data/project/agaid/h.noorazar/sidFabio_FV/"
data_dir <- paste0(database, "00_cumGDD_separateLocationsModels/", 
                   veg_type, "_special_order/", gsub("-", "", model_type), "/")

out_dir_base = database
param_dir = paste0(database, "/000_parameters/")

######################################################################

VIC_grids = data.table(read.csv(paste0(param_dir, "tomato_crd_trial.csv")))
VIC_grids = VIC_grids[VIC_grids$CRD %in% c("CA40", "CA50", "CA51", "FL50", "MI80")]
grid_count = dim(VIC_grids)[1]

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params.csv"),  as.is=T))
veg_params=veg_params[veg_params$veg==veg_type, ]


fabio_future_close_startDoY <- data.table(read.csv(paste0(param_dir, "fabio_future_close_startDoY.csv"),  as.is=T))

# 3. Process the data -----------------------------------------------------
col_names <- c("location", "year", "days_to_maturity", 
               "no_days_in_opt_interval", "no_of_extreme_cold", "no_of_extreme_heat",
               "cum_solar")

# 36 below comes from 1980-2015.
if (model_type=="observed"){
  year_count=2015-1980+1
  days_to_maturity_tb <- setNames(data.table(matrix(nrow = grid_count*year_count, 
                                                    ncol = length(col_names))), col_names)
  days_to_maturity_tb$location <- rep.int(VIC_grids$location, year_count)
  
  setorderv(days_to_maturity_tb,  ("location"))
  days_to_maturity_tb$year <- rep.int(c(1980:2015), grid_count)
} else{
  # 2041-2070
  year_count = 2070-2041+1
  days_to_maturity_tb <- setNames(data.table(matrix(nrow = grid_count*year_count, 
                                                    ncol = length(col_names))), col_names)
  days_to_maturity_tb$location <- rep.int(VIC_grids$location, year_count)
  setorderv(days_to_maturity_tb,  ("location"))
  days_to_maturity_tb$year <- rep.int(c(2041:2070), grid_count)
}
# print ("line 65")
# print (days_to_maturity_tb)
# x <- sapply(VIC_grids, 
#             function(x) strsplit(x, "_")[[1]], 
#             USE.NAMES=FALSE)
# lat = x[2, ]
# long = x[3, ]
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

  if (file.exists(paste0(data_dir, fileName, ".csv"))){
    # The following function will look into the right directory when 
    data_tb <- data.table(read.csv(paste0(data_dir, fileName, ".csv")))
    data_tb$row_num <- seq.int(nrow(data_tb))
    #
    # Here is where we have filtered 2041 to 2070: unique(days_to_maturity_tb$year)
    #
    for (a_year in sort(unique(days_to_maturity_tb$year))){
      curr_row_num <- data_tb[data_tb$year==a_year & data_tb$doy==start_doy, ]$row_num
      curr_data <- data_tb[data_tb$row_num>=curr_row_num, ]
      curr_data <- data.table(curr_data)
      curr_data$cum_GDD <- 0
      curr_data[, cum_GDD := cumsum(daily_GDD)] # , by=list(year)
      
      curr_data$cum_SRAD <- 0
      curr_data[, cum_SRAD := cumsum(SRAD)] # , by=list(year)

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
      # print ("107")
      # print (fileName)
      # print (a_year)
      # print (tail(start_to_maturity_tb, 1)$cum_SRAD)
      # print (days_to_maturity_tb)
      # days_to_maturity_tb$cum_solar<-tail(start_to_maturity_tb, 1)$cum_SRAD
      days_to_maturity_tb$cum_solar[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year]=tail(start_to_maturity_tb, 1)$cum_SRAD
      # print (start_to_maturity_tb)
      # print (days_to_maturity_tb[days_to_maturity_tb$location==fileName & days_to_maturity_tb$year==a_year])
      # print ("_113_____________________________________")


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
}

current_out = paste0(out_dir_base, "/01_countDays_toReachMaturity/", veg_type, "_special_order/start_doy_", start_doy, "/") # "_", model_type, 
if (dir.exists(current_out) == F) {
    dir.create(path = current_out, recursive = T)
}

write.csv(days_to_maturity_tb, 
          file = paste0(current_out, gsub("-", "", model_type), "_start_DoY_", start_doy, "_days2maturity_EE.csv"), 
          row.names=FALSE)



# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)
