#
# Sept. 21. 2022
#
# .libPaths("/data/hydro/R_libs35")
# .libPaths()
library(data.table)
source_path = "/home/h.noorazar/Sid/sidFabio/SidFabio_core.R"
source(source_path)
options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

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
database <- "/data/project/agaid/h.noorazar/sidFabio_FV/"
out_database = database

data_dir <- paste0(database, "00_cumGDD_separateLocationsModels/", 
                   veg_type, "/", model_type, "_sBs_", param_type, "_Oct24/")

param_dir = paste0(database, "/000_parameters/")

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params.csv"),  as.is=T))
veg_params=veg_params[veg_params$veg==veg_type, ]

file_list_name = "VIC_noPasture_CRD_ID_unique.csv"
VIC_grids <- data.table(read.csv(paste0(param_dir, file_list_name)))

# Pick up california
VIC_grids <- VIC_grids %>%
             filter(STASD_N %in% c(640, 650, 651, 680))%>%
             data.table()

grid_count = dim(VIC_grids)[1]
print (paste0("line 50: ", grid_count))
# 3. Process the data -----------------------------------------------------
col_names <- c("file_name", "year", "days_to_maturity", 
               "no_of_extreme_cold", "no_of_extreme_heat",
               "linear_cumGDD", "NL_cumGDD", "cum_solar")

# 36 below comes from 1980-2015.
if (model_type=="observed"){
  start_yr=1980
  end_yr=2015
 } else{
  start_yr=2041
  end_yr=2070 
}

year_count = end_yr-start_yr+1
days_to_maturity_tb <- setNames(data.table(matrix(nrow = grid_count*year_count, 
                                                  ncol = length(col_names))), col_names)
days_to_maturity_tb$file_name <- rep.int(VIC_grids$file_name, year_count)
setorderv(days_to_maturity_tb,  ("file_name"))
days_to_maturity_tb$year <- rep.int(c(start_yr:end_yr), grid_count)
print ("line 71: ")
print (dim(days_to_maturity_tb))
print (length(unique(days_to_maturity_tb$file_name)))
print ("line 74: ")
print (head(VIC_grids, 2))


counter=1
for(fileName in VIC_grids$file_name){

  # Look into the right directory
  if (file.exists(paste0(data_dir, fileName, ".csv"))){
    data_tb <- data.table(read.csv(paste0(data_dir, fileName, ".csv")))
    if (counter<20){
      print ("line 81, read data_tb: ")
      print (head(data_tb, 2))
    }
    
    data_tb$row_num <- seq.int(nrow(data_tb))

    # add day of year
    data_tb$doy = 1
    data_tb[, doy := cumsum(doy), by=list(year)]
    #
    # Here is where we have filtered 2041 to 2070: unique(days_to_maturity_tb$year) 
    # using start_yr and end_yr defined above.
    #
    for (a_year in sort(unique(days_to_maturity_tb$year))){
      curr_row_num <- data_tb[data_tb$year==a_year & data_tb$doy==start_doy, ]$row_num
      curr_data <- data_tb[data_tb$row_num>=curr_row_num, ]
      curr_data <- data.table(curr_data)
      curr_data$linear_cumGDD <- 0
      curr_data[, linear_cumGDD := cumsum(linear_daily_GDD)] # , by=list(year)
      if (counter<20){
        print ("line 103, curr_data: ")
        print (head(curr_data, 2))
      }

      curr_data$NL_cumGDD <- 0
      curr_data[, NL_cumGDD := cumsum(WEDD)] # , by=list(year)

      curr_data$cum_SRAD <- 0
      curr_data[, cum_SRAD := cumsum(SRAD)] # , by=list(year)

      day_of_maturity=curr_data[curr_data$linear_cumGDD >= veg_params[veg_params$veg==veg_type]$maturity_gdd]
      dayCount = day_of_maturity$row_num[1]-curr_data$row_num[1]

      # Record days_to_maturity
      days_to_maturity_tb$days_to_maturity[days_to_maturity_tb$file_name==fileName & days_to_maturity_tb$year==a_year] = dayCount

      days_to_maturity_tb$linear_cumGDD[days_to_maturity_tb$file_name==fileName & days_to_maturity_tb$year==a_year] = day_of_maturity[1, ]$linear_cumGDD
      days_to_maturity_tb$NL_cumGDD[days_to_maturity_tb$file_name==fileName & days_to_maturity_tb$year==a_year] = day_of_maturity[1, ]$NL_cumGDD

      # Subset the table between start day and day of maturity to count
      # the number of days in optimum interval and extreme events
      start_to_maturity_tb = curr_data %>%
                             filter(row_num>=curr_data$row_num[1]) %>%
                             filter(row_num<=day_of_maturity$row_num[1]) %>%
                             data.table()

      days_to_maturity_tb$cum_solar[days_to_maturity_tb$file_name==fileName & days_to_maturity_tb$year==a_year]=tail(start_to_maturity_tb, 1)$cum_SRAD

      extreme_cold_tb = start_to_maturity_tb %>%
                        filter(tmin<=veg_params$cold_stress) %>%
                        data.table()

      days_to_maturity_tb$no_of_extreme_cold[days_to_maturity_tb$file_name==fileName & days_to_maturity_tb$year==a_year] = dim(extreme_cold_tb)[1]


      extreme_heat_tb = start_to_maturity_tb %>%
                        filter(tmax>=veg_params$heat_stress) %>%
                        data.table()

      days_to_maturity_tb$no_of_extreme_heat[days_to_maturity_tb$file_name==fileName & days_to_maturity_tb$year==a_year] = dim(extreme_heat_tb)[1]
    }
  }
}

current_out = paste0(out_database, "/01_countDays_2Maturity_sBs/", 
                     veg_type, "_", param_type, "/") 

if (dir.exists(current_out) == F) {dir.create(path = current_out, recursive = T)}

print ("line 160")
print (head(days_to_maturity_tb, 10))

write.csv(days_to_maturity_tb, 
          file = paste0(current_out, model_type, "_start_DoY_", start_doy, "_days2maturity_EE.csv"), 
          row.names=FALSE)

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)



