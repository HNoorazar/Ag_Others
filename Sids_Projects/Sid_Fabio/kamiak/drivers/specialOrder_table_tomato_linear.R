


##################################
##
##  Sept 7. Table for Result section of overleaf file. 
##  This is about Email of Fabio on Aug. 30: special order where each county has its own unique
##  start DoY!
##
rm(list=ls())
library(data.table)
library(dplyr)
library(stringr)

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

################################
########
######## Directories
########

data_dir_base = "/Users/hn/Documents/01_research_data/Sid/SidFabio/"
data_dir = paste0(data_dir_base, "/02_aggregate_Maturiry_EE_linear_Kamiak/", veg_type, "/special_order_Aug_30_email/")

out_dir = paste0(data_dir_base, "03_table_results/", veg_type, "/")
if (dir.exists(out_dir) == F) {
    dir.create(path = out_dir, recursive = T)
}

file_names = c("annual_means_within_CRD_properSpecialDoY.csv", "within_TP_median_of_annual_means_within_CRD_properSpecialDoY.csv")


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

# annual_means_within_CRD <- dplyr::left_join(x=annual_means_within_CRD, y=alphabet_CRD, by="STASD_N")

# within_TP_median_of_annual_means_within_CRD <- dplyr::left_join(x=within_TP_median_of_annual_means_within_CRD, 
#                                                                 y=alphabet_CRD, by="STASD_N")


annual_means_within_CRD$CRD <- factor(annual_means_within_CRD$CRD, 
                                      levels=chosen_CRD_alph, order=TRUE)

within_TP_median_of_annual_means_within_CRD$CRD <- factor(within_TP_median_of_annual_means_within_CRD$CRD, 
                                                          levels = chosen_CRD_alph, order=TRUE)

start_DOY = sort(unique(annual_means_within_CRD$startDoY))
annual_means_within_CRD$startDoY <- factor(annual_means_within_CRD$startDoY, 
                                           levels = start_DOY, order=TRUE)
within_TP_median_of_annual_means_within_CRD$startDoY <- factor(within_TP_median_of_annual_means_within_CRD$startDoY, 
                                                               levels=start_DOY, order=TRUE)

median_of_means_3col <- within_TP_median_of_annual_means_within_CRD[, c("CRD", "median_of_mean_days_to_maturity", "startDoY")]

# col_names <- as.character(sort(as.numeric(array(unique(median_of_means_3col$startDoY)))))
# col_names <- c("CRD", col_names)
# table_to_export <- setNames(data.table(matrix(nrow=length(chosen_CRD_alph), 
#                                               ncol = length(col_names))), 
#                             col_names)

# table_to_export$CRD <- chosen_CRD_alph

# for (a_col in unique(median_of_means_3col$startDoY)){
#   for (a_crd in chosen_CRD_alph){
#   	curr_median=round(median_of_means_3col[median_of_means_3col$CRD==a_crd & startDoY==a_col]$median_of_mean_days_to_maturity)

#     table_to_export[a_col]
#   }
# }

table_to_export = reshape(median_of_means_3col, idvar="startDoY", timevar="CRD", direction="wide")
table_to_export <- table_to_export[order(startDoY, ),]
cols_to_round <- names(table_to_export)[2:length(names(table_to_export))]

# round the numbers
table_to_export[,(cols_to_round) := round(.SD,0), .SDcols=cols_to_round]

# change the column names. They are long, but descriptive:
x <- sapply(colnames(table_to_export)[2: length(colnames(table_to_export))], 
            function(x) strsplit(x, "\\.")[[1]], 
            USE.NAMES=FALSE)
new_colNames = x[2, ]

setnames(table_to_export, 
         old=colnames(table_to_export)[2: length(colnames(table_to_export))], 
         new=new_colNames)


# change the order of columns
table_to_export = table_to_export[, c(1, 4, 5, 3, 2, 6)]

write.csv(table_to_export, 
          file = paste0(out_dir, "median_of_means_of_days_to_maturity_wide_specialOrder_Aug30Email.csv"), 
          row.names=FALSE)


######### SR 


median_of_means_3col <- within_TP_median_of_annual_means_within_CRD[, c("CRD", "median_of_mean_of_cum_solar", "startDoY")]

table_to_export = reshape(median_of_means_3col, idvar="startDoY", timevar="CRD", direction="wide")
table_to_export <- table_to_export[order(startDoY, ),]
cols_to_round <- names(table_to_export)[2:length(names(table_to_export))]

# round the numbers
table_to_export[,(cols_to_round) := round(.SD,0), .SDcols=cols_to_round]

# change the column names. They are long, but descriptive:
x <- sapply(colnames(table_to_export)[2: length(colnames(table_to_export))], 
            function(x) strsplit(x, "\\.")[[1]], 
            USE.NAMES=FALSE)
new_colNames = x[2, ]

setnames(table_to_export, 
         old=colnames(table_to_export)[2: length(colnames(table_to_export))], 
         new=new_colNames)


# change the order of columns
table_to_export = table_to_export[, c(1, 4, 5, 3, 2, 6)]

write.csv(table_to_export, 
          file = paste0(out_dir, "median_of_means_of_cumSR_wide_specialOrder_Aug30Email.csv"), 
          row.names=FALSE)



