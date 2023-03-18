###############################################################
## 
##  Sept 22nd
##  2050s. and observed. Fabio had some plots and wanted to 
##  add future to the plots.
##
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

tomato_crd_trial = read.csv(paste0(param_dir, "tomato_crd_trial.csv"))
tomato_crd_trial <- within(tomato_crd_trial, remove(location))
tomato_crd_trial<-data.table(tomato_crd_trial)
tomato_crd_trial <- unique(tomato_crd_trial)
################################
########
######## Directories
########

data_dir_base = "/Users/hn/Documents/01_research_data/Sid/SidFabio/"
sub_dirs = c("tomato_linear", "tomato_nonlinear_claudio", "tomato_nonlinear_fabio")

file_names = c("annual_means_within_CRD.csv", "within_TP_median_of_annual_means_within_CRD.csv")

################################
################################
########
########  TS Plots
########

col_to_plot="mean_days_to_maturity"
for (a_subdir in sub_dirs){
  out_dir = paste0(data_dir_base, "03_plots/", a_subdir, "/")
  if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}

  data_dir = paste0(data_dir_base, "/02_aggregate_Maturiry_EE_Kamiak/", a_subdir, "/")
  ########
  ########   read data
  ########

  annual_means_within_CRD = data.table(read.csv(paste0(data_dir, file_names[1])))
  within_TP_median_of_annual_means_within_CRD = data.table(read.csv(paste0(data_dir, file_names[2])))

  annual_means_within_CRD <- dplyr::left_join(x=annual_means_within_CRD, 
                                              y=tomato_crd_trial, by="STASD_N")

  within_TP_median_of_annual_means_within_CRD <- dplyr::left_join(x=within_TP_median_of_annual_means_within_CRD, 
                                                                  y=tomato_crd_trial, by="STASD_N")

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

  time_p <- c("observed", "2050s")

  annual_means_within_CRD$time_period <- factor(annual_means_within_CRD$time_period, 
                                                levels =time_p, order=TRUE)
  within_TP_median_of_annual_means_within_CRD$time_period <- factor(within_TP_median_of_annual_means_within_CRD$time_period, 
                                                                    levels=time_p, order=TRUE)

  mean_days_to_maturity_plot<- annual_TS_3(d1=annual_means_within_CRD, y_colname=col_to_plot, 
                                           fil="time_period",
                                           y_label="average number of days to reach maturity")

  fName = paste(col_to_plot, veg_type, sep = "_")
  ggsave(plot = mean_days_to_maturity_plot,
         filename = paste0(fName, ".png"), 
         width=30, height=21, units = "in", 
         dpi=200, device = "png",
         path=out_dir,
         limitsize = FALSE)

  box_annual_means_within_CRD=box_annual_startDoY_x(dt=annual_means_within_CRD, y_colname=col_to_plot, 
                                                    title_="week of Sep. 20, Fabio-Sid Brain Storm",
                                                    yLab="average number of days to maturity")
  # param_type="fabio"
  fName = paste("box_mean_days_to_maturity", veg_type, sep = "_")
  ggsave(plot = box_annual_means_within_CRD,
         filename = paste0(fName, ".png"), 
         width=7, height=8, units = "in", 
         dpi=200, device = "png",
         path=out_dir,
         limitsize = FALSE)


  for (a_crd in chosen_CRD_alph){
    annual_means_within_CRD_subset <- annual_means_within_CRD %>%
                                      filter(CRD==a_crd) %>% data.table()
    
    mean_days_to_maturity_plot<- annual_TS_3(d1=annual_means_within_CRD_subset, y_colname=col_to_plot, 
                                             fil="time_period",
                                             y_label="average number of days to reach maturity")

    subset_out_dir = paste0(out_dir, "mean_days_to_maturity_separate_CRD/")
    if (dir.exists(subset_out_dir) == F) {dir.create(path=subset_out_dir, recursive = T)}

    fName = paste(col_to_plot, veg_type, a_crd, sep = "_")
    ggsave(plot = mean_days_to_maturity_plot,
           filename = paste0(fName, ".png"), 
           width=20, height=10, units = "in", 
           dpi=200, device = "png",
           path=subset_out_dir,
           limitsize = FALSE)
    }
}


######## Useless
# for (a_start_DOY in start_DOY){
#   annual_means_within_CRD_subset <- annual_means_within_CRD %>%
#                   filter(startDoY==a_start_DOY) %>%
#                   data.table()

#   mean_days_to_maturity_plot<- annual_TS_timePeriod_groupColor(d1=annual_means_within_CRD_subset, 
#                      colname="mean_days_to_maturity", fil="maturity age")
 
#   fName = paste(col_to_plot, veg_type, a_start_DOY, sep = "_")

#   subset_out_dir = paste0(out_dir, "mean_days_to_maturity_separate_start_DoY/")
#  if (dir.exists(subset_out_dir) == F) {
#    dir.create(path=subset_out_dir, recursive = T)
#   }
#   ggsave(plot = mean_days_to_maturity_plot,
#      filename = paste0(fName, ".png"), 
#      width=20, height=7.5, units = "in",
#      dpi=200, device = "png",
#      path=subset_out_dir,
#      limitsize = FALSE)
# }








