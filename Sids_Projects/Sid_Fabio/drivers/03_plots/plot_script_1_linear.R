###############################################################
## 
##  Aug. 25th. 
##  First attempt to look at the aggregated files.
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

veg_type = "tomato"  # "tomato" at this point, later: "carrot", "spinach", "strawberry", "tomato"
# param_type = "fabio" # "fabio" or "claudio"

################################
########
######## Directories
########

data_dir_base = "/Users/hn/Documents/01_research_data/Sid/SidFabio/"
data_dir = paste0(data_dir_base, "/02_aggregate_Maturiry_EE_linear/", veg_type, "/")


out_dir = paste0(data_dir_base, "03_plots/", veg_type, "/")
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
########    TS Plots
########


mean_days_to_maturity_plot<- annual_TS(d1=annual_means_within_CRD, 
                                       colname="mean_days_to_maturity", 
                                       fil="maturity age")

fName = paste("mean_days_to_maturity", veg_type, sep = "_")

# ggsave(plot = mean_days_to_maturity_plot,
#        filename = paste0(fName, ".png"), 
#        width=50, height=50, units = "in", 
#        dpi=200, device = "png",
#        path=out_dir,
#        limitsize = FALSE)


ggsave(plot = mean_days_to_maturity_plot,
       filename = paste0(fName, ".png"), 
       width=50, height=30, units = "in", 
       dpi=200, device = "png",
       path=out_dir,
       limitsize = FALSE)



for (a_crd in chosen_CRD_alph){
  annual_means_within_CRD_subset <- annual_means_within_CRD %>%
                                    filter(CRD==a_crd) %>%
                                    data.table()
 mean_days_to_maturity_plot<- annual_TS(d1=annual_means_within_CRD_subset, 
                                        colname="mean_days_to_maturity", 
                                        fil="maturity age")
 
 fName = paste("mean_days_to_maturity", veg_type, a_crd, sep = "_")

 subset_out_dir = paste0(out_dir, "mean_days_to_maturity_separate_CRD/")
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


for (a_start_DOY in start_DOY){
  annual_means_within_CRD_subset <- annual_means_within_CRD %>%
                                    filter(startDoY==a_start_DOY) %>%
                                    data.table()

  mean_days_to_maturity_plot<- annual_TS(d1=annual_means_within_CRD_subset, 
                                         colname="mean_days_to_maturity", fil="maturity age")
 
  fName = paste("mean_days_to_maturity", veg_type, a_start_DOY, sep = "_")

  subset_out_dir = paste0(out_dir, "mean_days_to_maturity_separate_start_DoY/")
   if (dir.exists(subset_out_dir) == F) {
       dir.create(path=subset_out_dir, recursive = T)
    }
  ggsave(plot = mean_days_to_maturity_plot,
         filename = paste0(fName, ".png"), 
         width=20, height=7.5, units = "in",
         dpi=200, device = "png",
         path=subset_out_dir,
         limitsize = FALSE)
}


################################
########
########    Box Plots
########
box_annual_means_within_CRD=box_annual_startDoY_x(dt=annual_means_within_CRD, 
                                                  colname="mean_days_to_maturity", 
                                                  time_period="observed")

# param_type="fabio"
out_dir = paste0(data_dir_base, "03_plots/", veg_type, "/")
if (dir.exists(out_dir) == F) {
    dir.create(path = out_dir, recursive = T)
}
fName = paste("box_mean_days_to_maturity", veg_type, sep = "_")
ggsave(plot = box_annual_means_within_CRD,
       filename = paste0(fName, ".png"), 
       width=17, height=4, units = "in", 
       dpi=200, device = "png",
       path=out_dir,
       limitsize = FALSE)



