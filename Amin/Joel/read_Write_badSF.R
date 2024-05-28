
rm(list=ls())
library(data.table)
# library(rgdal) There is no rgdal anymore.
library(dplyr)
library(sp)
library(sf)
library(foreign)
options(digits=9)
options(digit=9)
################################################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/Amin/Joel/")


################################################################################
years <- seq(from = 2021, to = 2023, by = 1)


WSDA <- read_sf(paste0(data_dir, "From_Joel/WSDA2023DoubleCropOnly_May2024/", "WSDA2023DoubleCropOnly.shp"))

gdb <- path.expand(paste0(data_dir, "From_Joel/WSDA2023DoubleCropOnly_May2024/"))
# WSDA_V2 <- readOGR(gdb, "WSDACrop_2015")

fc <- sf::st_read(paste0(data_dir, "From_Joel/WSDA2023DoubleCropOnly_May2024/"), layer = "gdb")
fc <- sf::st_read(paste0(data_dir, "From_Joel/WSDA2023DoubleCropOnly_May2024/"), layer = "a00000001")
fc <- sf::st_read(paste0(data_dir, "From_Joel/WSDA2023DoubleCropOnly_May2024/"), layer = "a0000000")
fc <- sf::st_read(paste0(data_dir, "From_Joel/WSDA2023DoubleCropOnly_May2024/"), layer = "a0000000a")


for (yr in years){
  WSDA <- readOGR(paste0(data_dir, "WSDACrop_", yr, "/WSDACrop_", yr, ".shp"),
                  layer = paste0("WSDACrop_", yr), 
                  GDAL1_integer64_policy = TRUE)
  WSDA <- WSDA@data

  # WSDA$Irrigtn <- tolower(WSDA$Irrigtn)
  # WSDA$Irrigtn <- as.character(WSDA$Irrigtn)
  # WSDA$Irrigtn[is.na(WSDA$Irrigtn)] <- "na"
  
  typess <- c("total_acr_", "Irr_acr_", "NonIrr_acr_", "total_dbl_acr_", "Irr_Dbl_acr_", "NonIrr_Dbl_acr_")
  typess <- paste0(typess, yr)
  counties <- sort(unique(WSDA$county))

  output_tbl <- data.table(matrix(666, nrow = length(counties), ncol = (1+length(typess)) ))
  setnames(output_tbl, old=colnames(output_tbl), new = c("county", typess) )
  output_tbl$county <- counties

  for (row in c(1:nrow(output_tbl))){
   curr <- WSDA %>% filter(county == output_tbl[row]$county)

    Irr_tbl <- filter_out_non_irrigated_datatable(curr)

    total_acr <- sum(curr$ExctAcr)
    Irr_acr <- sum(Irr_tbl$ExctAcr)
    NonIrr_acr <- total_acr - Irr_acr

    curr_dbl <- filter_double_by_Notes(curr)
    curr_dbl_Irr <- filter_out_non_irrigated_datatable(curr_dbl)

    total_dbl_acr <- sum(curr_dbl$ExctAcr)
    Irr_dbl_acr <- sum(curr_dbl$ExctAcr)
    NonIrr_dbl_acr <- total_dbl_acr - Irr_dbl_acr
    
    row_vals <- c(total_acr, Irr_acr, NonIrr_acr, total_dbl_acr, Irr_dbl_acr, NonIrr_dbl_acr)
    
    output_tbl[row, 2] <- row_vals[1]
    output_tbl[row, 3] <- row_vals[2]
    output_tbl[row, 4] <- row_vals[3]
    output_tbl[row, 5] <- row_vals[4]
    output_tbl[row, 6] <- row_vals[5]
    output_tbl[row, 7] <- row_vals[6]
  }

  cols <- names(output_tbl)[2:dim(output_tbl)[2]]
  output_tbl[,(cols) := round(.SD, 2), .SDcols=cols]

  main_out <- "/Users/hn/Documents/01_research_data/remote_sensing/Irr_NonIrr_Acr_perCounty/"
  if (dir.exists(main_out) == F) {dir.create(path = main_out, recursive = T)}
  
  out_name <- paste0("Irr_NonIrr_Acr_perCounty_", yr, ".csv")
  write.csv(output_tbl, paste0(main_out, out_name), row.names = F)
}

