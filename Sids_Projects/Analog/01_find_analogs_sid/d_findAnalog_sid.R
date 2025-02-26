.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(FNN)
library(EnvStats, lib.loc = "~/.local/lib/R3.5.1")

source_path = "/Users/hn/Documents/00_GitHub/Ag/Sids_Projects/Analog/core_analog_sid.R"
source_path = "/home/hnoorazar/sid/analog_pairwise_2021/core_analog_sid.R"
source(source_path)
options(digit=9)
options(digits=9)

################################################################################
################################################################################
# 
#                   Terminal arguments and parameters
# 
################################################################################
args = commandArgs(trailingOnly=TRUE)

emission = args[1] # rcp45 rcp85
# time_period = args[2]

print (paste0("emission is: ", emission))
# print (paste0("time_period is: ", time_period))
################################################################################
# 
#                     set up proper directories
# 
################################################################################
in_dir <- "/Users/hn/Documents/01_research_data/Sids_Projects/analog_data/"
in_dir <- "/data/hydro/users/Hossein/sids_projects/analog/00_data_from_sid/"

main_out <- "/data/hydro/users/Hossein/sids_projects/analog/"
out_dir <- paste0(main_out, "01_analogs_perLocModel/")

print (paste0("out_dir is : ",  out_dir))
if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }

################################################################################
# 
#                   Read Data
# 
################################################################################

all_dt <- data.table(read.csv(paste0(in_dir, "Climate_Data_All_Variables.csv"), as.is=TRUE))


#
#  future_dt includes the locations for which we are looking for an analog for.
#
future_data <- all_dt %>%
               filter(scenario == emission) %>%
               data.table()

#
#  all_dt_usa will be the historical data that can come from all the USA
#
all_dt_usa <- all_dt %>%
              filter(scenario == "gridmet") %>%
              data.table()

all_dt_usa <- tidyr::drop_na(all_dt_usa)



future_data <- read.csv(paste0(in_dir, "Bmatrix_RCP4.5_20_50.csv"))
all_dt_usa <- read.csv(paste0(in_dir, "Amatrix_90_10_v1.csv"))

colnames(all_dt_usa) <- tolower(colnames(all_dt_usa))
colnames(future_data) <- tolower(colnames(future_data))

all_dt_usa$location <- paste0(all_dt_usa$lat, "_", all_dt_usa$lon)
future_data$location <- paste0(future_data$lat, "_", future_data$lon)

n_nghs <- nrow(all_dt_usa)
n_nghs <- n_nghs - 30


################################################################################
start_time <- Sys.time()


non_numeric_cols <- colnames(all_dt_usa)[c(1, 2, 28)]
numeric_col <- colnames(all_dt_usa)[4:28]
print (non_numeric_cols)
#
# I will add two extra for-loops outside the most
# interior for-loop and try if it works out, before making 100 jobs!!!!
#

# it seems this for-loop is already in the function find_NN_info_biofix(.)
# moreover, some locations have missing models! I do not know why! so,
# the second loop picks the model that are present for a given location!
# 
for (a_future_loc in unique(future_data$location)){ 
  a_loc_future <- future_data %>%
                  filter(location == a_future_loc) %>%
                  # filter(model == a_model) %>%
                  data.table()
  ##  *** Sid ***
  ##  Read this Sid: it seems you have only 1 year for future data.
  ##  So, I add the following line so that the function find_NN_info_biofix(.) inside core module
  ##  does not break down; it uses the 'year' column of future data since in the past we had several years
  ##  for future data
  ##
  a_loc_future$year = c(1:nrow(a_loc_future))
  
  source_path = "/Users/hn/Documents/00_GitHub/Ag/Sids_Projects/Analog/Sep_27_2021/core_analog_sid.R"
  source(source_path)
  information = find_NN_info_biofix(ICV = all_dt_usa,
                                    historical_dt = all_dt_usa,
                                    future_dt = a_loc_future,
                                    n_neighbors = n_nghs,
                                    numeric_cols = numeric_col,
                                    non_numeric_cols = non_numeric_cols)

  NN_dist_tb = information[[1]]
  NN_loc_year_tb = information[[2]]
  NN_sigma_tb = information[[3]]

  saveRDS(NN_dist_tb,     paste0(out_dir, paste("/NN_dist_tb",     
                                                 a_future_loc, gsub("-", "_", a_model), time, emission, sep="_"), ".rds"))

  saveRDS(NN_loc_year_tb, paste0(out_dir, paste("/NN_loc_year_tb", 
                                                 a_future_loc, gsub("-", "_", a_model), time, emission, sep="_"), ".rds"))

  saveRDS(NN_sigma_tb,    paste0(out_dir, paste("/NN_sigma_tb", 
                                                     a_future_loc, gsub("-", "_", a_model), time, emission, sep="_"), ".rds"))
}



end_time <- Sys.time()
print( end_time - start_time)

