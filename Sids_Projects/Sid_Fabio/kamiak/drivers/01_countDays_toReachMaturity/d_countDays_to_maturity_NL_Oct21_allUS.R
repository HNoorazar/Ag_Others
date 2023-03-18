##
## Oct. 21. 2022
##
## d_countDays_to_maturity_NL_Oct21_allUS.R and associated files are 
## about the talk that Fabio has had with Claudio.
## He took average of medians of accumulated GDD over 104 days
## from planting dated where planting dates were march 15, .... etc.
## He does not believe there is a reason for Claudio to change
## this maturity GDDs.
##


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
database <- "/data/project/agaid/h.noorazar/sidFabio_FV/"
data_dir <- paste0(database, "00_cumGDD_separateLocationsModels/", veg_type, "/", 
                   model_type, "_nonLinear_", param_type, "_Oct21_2022_allUS/")

out_database = database
param_dir = paste0(database, "000_parameters/")

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params_Oct17_2022.csv"),  as.is=T))
veg_params=veg_params[veg_params$veg==veg_type, ]


VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique.csv")))
# VIC_grids = VIC_grids[VIC_grids$CRD %in% c("CA40", "CA50", "CA51", "FL50", "MI80")]

VIC_grids$location= paste0(VIC_grids$lat, "_" , VIC_grids$long)

grid_count = dim(VIC_grids)[1]

# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()

col_names <- c("location", "year", "days_to_maturity", 
               "no_of_extreme_cold", "no_of_extreme_heat", "cum_solar") # no_days_in_opt_interval

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
days_to_maturity_tb$location <- rep.int(VIC_grids$file_name, year_count)
setorderv(days_to_maturity_tb,  ("location"))
days_to_maturity_tb$year <- rep.int(c(start_yr:end_yr), grid_count)
counter=1

if (dir.exists(data_dir)){
  for(a_file_name in VIC_grids$file_name){
    # if (counter==1){
    #   print (paste0(data_dir, ", ", a_file_name))
    #   counter+=1
    # }
    # Look into the right directory
    if (file.exists(paste0(data_dir, a_file_name, ".csv"))){
      data_tb <- data.table(read.csv(paste0(data_dir, a_file_name, ".csv")))
      data_tb$row_num <- seq.int(nrow(data_tb))

      # add day of year
      data_tb$doy = 1
      data_tb[, doy := cumsum(doy), by=list(year)]
      #
      # Here is where we have filtered 2041 to 2070: unique(days_to_maturity_tb$year)
      #
      for (a_year in sort(unique(days_to_maturity_tb$year))){
        curr_row_num <- data_tb[data_tb$year==a_year & data_tb$doy==start_doy, ]$row_num
        curr_data <- data_tb[data_tb$row_num>=curr_row_num, ]
        curr_data <- data.table(curr_data)
        curr_data$cum_GDD <- 0
        curr_data[, cum_GDD := cumsum(WEDD)] # , by=list(year)

        curr_data$cum_SRAD <- 0
        curr_data[, cum_SRAD := cumsum(SRAD)] # , by=list(year)

        day_of_maturity=curr_data[curr_data$cum_GDD >= veg_params[veg_params$veg==veg_type]$maturity_gdd]
        dayCount = day_of_maturity$row_num[1]-curr_data$row_num[1]

        # Record days_to_maturity
        days_to_maturity_tb$days_to_maturity[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year] = dayCount

        # Subset the table between start day and day of maturity to count
        # the number of days in optimum interval and extreme events
        start_to_maturity_tb = curr_data %>%
                               filter(row_num>=curr_data$row_num[1]) %>%
                               filter(row_num<=day_of_maturity$row_num[1]) %>%
                               data.table()

        days_to_maturity_tb$cum_solar[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year]=tail(start_to_maturity_tb, 1)$cum_SRAD

        # optimum_table = start_to_maturity_tb %>%
        #                 filter(Tavg>=veg_params$optimum_low) %>%
        #                 filter(Tavg<=veg_params$optimum_hi) %>%
        #                 data.table()

        # days_to_maturity_tb$no_days_in_opt_interval[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year] = dim(optimum_table)[1]

        extreme_cold_tb = start_to_maturity_tb %>%
                          filter(tmin<=veg_params$cold_stress) %>%
                          data.table()

        days_to_maturity_tb$no_of_extreme_cold[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year] = dim(extreme_cold_tb)[1]


        extreme_heat_tb = start_to_maturity_tb %>%
                          filter(tmax>=veg_params$heat_stress) %>%
                          data.table()

        days_to_maturity_tb$no_of_extreme_heat[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year] = dim(extreme_heat_tb)[1]

      }
    }
  }


  current_out = paste0(out_database, "/01_countDays_toReachMaturity/",  
                                       veg_type, "_NL_", param_type, "_Oct21_2022_allUS_", start_yr, "_", end_yr, "/") 

  if (dir.exists(current_out) == F) {
      dir.create(path = current_out, recursive = T)
  }

  write.csv(days_to_maturity_tb, 
            file = paste0(current_out, model_type, "_start_DoY_", start_doy, "_days2maturity_EE.csv"), 
            row.names=FALSE)
}
# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)
