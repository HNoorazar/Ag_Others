###############################################################
## 
##  Sept 8th 
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

file_names = c("annual_means_within_CRD_properSpecialDoY.csv", 
               "within_TP_median_of_annual_means_within_CRD_properSpecialDoY.csv")

################################
########
########     read data
########

annual_means_within_CRD = data.table(read.csv(paste0(data_dir, file_names[1])))
within_TP_median_of_annual_means_within_CRD = data.table(read.csv(paste0(data_dir, file_names[2])))
fabio_future_close_startDoY = data.table(read.csv(paste0(param_dir, "fabio_future_close_startDoY.csv")))


chosen_CRD_alph <- c("CA40", "CA50", "CA51", "FL50", "MI80")

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
########    TS Plots
########
col_to_plot="mean_of_cum_solar"
mean_solar_over_growth_period <- annual_TS_timePeriod_groupColor_specialOrder(d1=annual_means_within_CRD, 
                                                                              y_colname=col_to_plot, 
                                                                              fil="time_period",
                                                                              y_label="average cumulative SR over growth period")
fName = paste(col_to_plot, veg_type, sep = "_")
ggsave(plot = mean_solar_over_growth_period,
       filename = paste0(fName, ".png"), 
       width=15, height=7, units = "in", 
       dpi=200, device = "png",
       path=out_dir,
       limitsize = FALSE)

for (a_crd in chosen_CRD_alph){
  annual_means_within_CRD_subset <- annual_means_within_CRD %>%
                                    filter(CRD==a_crd) %>%
                                    data.table()
 mean_days_to_maturity_plot<- annual_TS_timePeriod_groupColor(d1=annual_means_within_CRD_subset, 
                                                              y_colname=col_to_plot, 
                                                              fil="maturity age",
                                                              y_label="mean of cumulative SR over maturity period")
 
 fName = paste(col_to_plot, veg_type, a_crd, sep = "_")

 subset_out_dir = paste0(out_dir, "mean_cumuSR_growthPeriod_separate_CRD/")
  if (dir.exists(subset_out_dir) == F) {
      dir.create(path=subset_out_dir, recursive = T)
   }
 ggsave(plot = mean_days_to_maturity_plot,
        filename = paste0(fName, ".png"), 
        width=20, height=10, units = "in", 
        dpi=200, device = "png",
        path=subset_out_dir,
        limitsize = FALSE)
}


### Useless
# for (a_start_DOY in start_DOY){
#   annual_means_within_CRD_subset <- annual_means_within_CRD %>%
#                                     filter(startDoY==a_start_DOY) %>%
#                                     data.table()

#   mean_days_to_maturity_plot<- annual_TS_timePeriod_groupColor(d1=annual_means_within_CRD_subset, 
#                                                                y_colname=col_to_plot, fil="maturity age")
 
#   fName = paste("mean_cumulative_solar", veg_type, a_start_DOY, sep = "_")

#   subset_out_dir = paste0(out_dir, "mean_cumuSR_growthPeriod_separate_start_DoY/")
#    if (dir.exists(subset_out_dir) == F) {
#        dir.create(path=subset_out_dir, recursive = T)
#     }
#   ggsave(plot = mean_days_to_maturity_plot,
#          filename = paste0(fName, ".png"), 
#          width=20, height=7.5, units = "in",
#          dpi=200, device = "png",
#          path=subset_out_dir,
#          limitsize = FALSE)
# }

################################
########
########    Box Plots
########
box_annual_means_within_CRD=box_annual_startDoY_x(dt=annual_means_within_CRD, 
                                                  y_colname=col_to_plot,
                                                  title_="average cumulative SR during growing: tomato, observed, Fabio Aug. 30 email",
                                                  yLab="average cumulative SR")

# param_type="fabio"
fName = paste("box_cumSR_", veg_type, sep = "_")
ggsave(plot = box_annual_means_within_CRD,
       filename = paste0(fName, ".png"), 
       width=17, height=4, units = "in", 
       dpi=200, device = "png",
       path=out_dir,
       limitsize = FALSE)



