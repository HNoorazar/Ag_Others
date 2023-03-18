####
#### This module generates #s in the following link ()
#### https://docs.google.com/document/d/18KX24FkL70_Xhxagwx9EBRWeQmz-Ud-iuTXqnf9YXnk/edit?usp=sharing
####

rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

##############################################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/")
write_dir <- paste0("/Users/hn/Documents/01_research_data/Sid_Kirti_AGU/")
##############################################################################

Grant <- readOGR(paste0(data_dir, "Grant2017/Grant2017.shp"),
                       layer = "Grant2017", 
                       GDAL1_integer64_policy = TRUE)

Grant <- Grant[Grant@data$DataSrc != "nass", ]
Grant <- filter_out_non_irrigated_shapefile(Grant)
Grant <- Grant[grepl('2017', Grant@data$LstSrvD), ]

Grant <- Grant[grepl('corn', Grant@data$CropTyp), ]

big_Grant <- Grant[Grant@data$Acres>50, ]

sweet_corn <- big_Grant[grepl('sweet', big_Grant@data$CropTyp), ]
corn_field <- big_Grant[grepl('field', big_Grant@data$CropTyp), ]

big_Grant_seed <- Grant[Grant@data$Acres>=9, ]
corn_seed <- big_Grant_seed[grepl('seed', big_Grant_seed@data$CropTyp), ]

dim(sweet_corn)
dim(corn_seed)
dim(corn_field)


sweet_corn <- sweet_corn[1:5, ]
corn_seed <- corn_seed[1:5, ]
corn_field <- corn_field[1:5, ]
corn_fields <- raster::bind(sweet_corn, corn_seed, corn_field)

writeOGR(obj = corn_fields,
         dsn = paste0(write_dir, "/", "Sid_Kirti_AGU_Corn_Grant_Irr_Surv"), 
         layer = "Sid_Kirti_AGU_Corn_Grant_Irr_Surv", 
         driver = "ESRI Shapefile")



