###############################################################
## 
##  Sept 6th 
##  2050s. and observed. Special order Fabio. Linear models and params.
##

rm(list=ls())

library(dplyr)
library(data.table)
library(ggplot2)


source_path = "/home/hnoorazar/Sid/sidFabio/SidFabio_core_plot.R"
source_path = "/Users/hn/Documents/00_GitHub/Ag/Sids_Projects/Sid_Fabio/SidFabio_core_plot.R"
source(source_path)
options(digits=9)


################################
########
######## parameters
########
param_dir = "/Users/hn/Documents/01_research_data/Sid/SidFabio/parameters/"
veg_type = "tomato"  # "tomato" at this point, later: "carrot", "spinach", "strawberry", "tomato"

################################
########
######## Directories
########

data_dir_base = "/Users/hn/Documents/01_research_data/Sid/SidFabio/"
data_dir = paste0(data_dir_base, "/02_aggregate_Maturiry_EE_linear_Kamiak/", veg_type, "/special_order_Aug_30_email/")


out_dir = paste0(data_dir_base, "03_plots/specialOrder_2050s", veg_type, "/")
if (dir.exists(out_dir) == F) {
    dir.create(path = out_dir, recursive = T)
}

file_names = c("annual_means_within_CRD.csv", "within_TP_median_of_annual_means_within_CRD.csv")


################################
########
########     read data
########

annual_means_within_CRD = data.table(read.csv(paste0(data_dir, file_names[1])))
within_TP_median_of_annual_means_within_CRD = data.table(read.csv(paste0(data_dir, file_names[2])))
fabio_future_close_startDoY = data.table(read.csv(paste0(param_dir, "fabio_future_close_startDoY.csv")))

################################
########
########    Subset
########

unique(annual_means_within_CRD$STASD_N)

# 12 is Florida and 26 is Michigan.
chosen_CRD <- c(640, 650, 651, 1250, 2680)
chosen_CRD_alph <- c("CA40", "CA50", "CA51", "FL50", "MI80")
annual_means_within_CRD <- annual_means_within_CRD %>%
                           filter(STASD_N %in% chosen_CRD) %>%
                           data.table()

within_TP_median_of_annual_means_within_CRD <- within_TP_median_of_annual_means_within_CRD %>%
                                               filter(STASD_N %in% chosen_CRD) %>%
                                               data.table()


alphabet_CRD <- data.table()
alphabet_CRD$STASD_N <- chosen_CRD
alphabet_CRD$CRD <- chosen_CRD_alph

annual_means_within_CRD <- dplyr::left_join(x=annual_means_within_CRD, y=alphabet_CRD, by="STASD_N")

within_TP_median_of_annual_means_within_CRD <- dplyr::left_join(x=within_TP_median_of_annual_means_within_CRD, 
                                                                y=alphabet_CRD, by="STASD_N")


annual_means_within_CRD$CRD <- factor(annual_means_within_CRD$CRD, 
                                      levels = chosen_CRD_alph, order=TRUE)

within_TP_median_of_annual_means_within_CRD$CRD <- factor(within_TP_median_of_annual_means_within_CRD$CRD, 
                                                          levels = chosen_CRD_alph, order=TRUE)

start_DOY = sort(unique(annual_means_within_CRD$startDoY))
annual_means_within_CRD$startDoY <- factor(annual_means_within_CRD$startDoY, 
                                           levels = start_DOY, order=TRUE)
within_TP_median_of_annual_means_within_CRD$startDoY <- factor(within_TP_median_of_annual_means_within_CRD$startDoY, 
                                                               levels=start_DOY, order=TRUE)

################################
########
########    Subset Each county with its proper start_DoY
########
annual_means_within_CRD_subset = data.table()
within_TP_median_of_annual_means_within_CRD_subset = data.table()

for (a_CRD in unique(annual_means_within_CRD$CRD)){

  observed <- annual_means_within_CRD %>% 
              filter(CRD==a_CRD) %>% 
              filter(startDoY %in% fabio_future_close_startDoY[fabio_future_close_startDoY$CRD==a_CRD, "observed_start_doy"])%>% 
              filter(time_period =="observed")%>% 
              data.table()
  
  future_2050s <- annual_means_within_CRD %>% 
                  filter(CRD==a_CRD) %>% 
                  filter(startDoY %in% fabio_future_close_startDoY[fabio_future_close_startDoY$CRD==a_CRD][1, 3:4])%>% 
                  filter(time_period =="2050s")%>% 
                  data.table()

  curr_annual_means_within_CRD_subset <- rbind(observed, future_2050s)

  observed <- within_TP_median_of_annual_means_within_CRD %>% 
              filter(CRD==a_CRD) %>% 
              filter(startDoY %in% fabio_future_close_startDoY[fabio_future_close_startDoY$CRD==a_CRD, "observed_start_doy"])%>% 
              filter(time_period =="observed")%>% 
              data.table()
  
  future_2050s <- within_TP_median_of_annual_means_within_CRD %>% 
                  filter(CRD==a_CRD) %>% 
                  filter(startDoY %in% fabio_future_close_startDoY[fabio_future_close_startDoY$CRD==a_CRD][1, 3:4])%>% 
                  filter(time_period =="2050s")%>% 
                  data.table()
  

  curr_within_TP_median_of_annual_means_within_CRD_subset <- rbind(observed, future_2050s)

  annual_means_within_CRD_subset = rbind(annual_means_within_CRD_subset, curr_annual_means_within_CRD_subset)
  within_TP_median_of_annual_means_within_CRD_subset = rbind(within_TP_median_of_annual_means_within_CRD_subset, 
                                                            curr_within_TP_median_of_annual_means_within_CRD_subset)
}
####
####   Update the datasets so that we do not have to change 
####   the rest of the code:
#### 
annual_means_within_CRD <- annual_means_within_CRD_subset
within_TP_median_of_annual_means_within_CRD <- within_TP_median_of_annual_means_within_CRD_subset



write.csv(annual_means_within_CRD, 
          file = paste0(data_dir, "annual_means_within_CRD_properSpecialDoY.csv"), 
          row.names=FALSE)

write.csv(within_TP_median_of_annual_means_within_CRD, 
          file = paste0(data_dir, "within_TP_median_of_annual_means_within_CRD_properSpecialDoY.csv"), 
          row.names=FALSE)


