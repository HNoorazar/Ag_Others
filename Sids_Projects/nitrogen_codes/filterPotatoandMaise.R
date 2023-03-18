rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)


options(digits=9)
options(digit=9)

SF_dir = "C://Users/kirti/Documents/Nitrogen_data/"
WSDA <- readOGR(paste0(SF_dir, "WSDA_2020/", "WSDA2020.shp"),
                 layer = "WSDA2020", 
                 GDAL1_integer64_policy = TRUE)


# add ID to each field
add_identifier <- function(SF, year){
  SF@data <- tibble::rowid_to_column(SF@data, "ID")
  SF@data$ID <- paste0(SF@data$ID, "_WSDA_SF_", year)
  return(SF)
}

WSDA = add_identifier(WSDA, 2020)

filter_out_non_irrigated_shapefile <- function(dt){
  dt@data$Irrigtn <- tolower(dt@data$Irrigtn)
  dt@data$Irrigtn[is.na(dt@data$Irrigtn)] <- "na"

  dt <- dt[!grepl('none', dt$Irrigtn), ] # toss out those with None in irrigation
  dt <- dt[!grepl('unknown', dt$Irrigtn), ] # toss out Unknown
  dt <- dt[!grepl('na', dt$Irrigtn), ] # toss out NAs
  return(dt)
}



# rename irrigation column so it is consisten with all my codes.
# abbreviations happen when R writes the shapefile to disk.
setnames(WSDA@data, "Irrigation", "Irrigtn")
WSDA@data$Irrigtn <- tolower(WSDA@data$Irrigtn)
WSDA =filter_out_non_irrigated_shapefile(WSDA)

setnames(WSDA@data, "LastSurvey", "LstSrvD")
WSDA <- WSDA[grepl(as.character(2020), WSDA@data$LstSrvD), ]

counties<- c("Adams", "Asotin", "Benton", "Chelan", "Columbia", "Cowlitz", "Douglas", 
             "Ferry", "Franklin", "Garfield", "Grant", "Kittitas", 
             "Klickitat", "Lincoln", "Okanogan", 
             "Pend Orielle", "Spokane", "Stevens", "Walla Walla", "Whitman", "Yakima")

pick_basin_counties <- function(sff){  
  if ("County" %in% colnames(sff@data)){
    Okanogan <- sff[grepl('Okanogan', sff$County), ]
    Chelan <- sff[grepl('Chelan', sff$County), ]

    Kittitas <- sff[grepl('Kittitas', sff$County), ]
    Yakima <- sff[grepl('Yakima', sff$County), ]

    Klickitat <- sff[grepl('Klickitat', sff$County), ]
    Douglas <- sff[grepl('Douglas', sff$County), ]

    Ferry <- sff[grepl('Ferry', sff$County), ]
    Lincoln <- sff[grepl('Lincoln', sff$County), ]

    Grant <- sff[grepl('Grant', sff$County), ]
    Benton <- sff[grepl('Benton', sff$County), ]

    Adams <- sff[grepl('Adams', sff$County), ]
    Franklin <- sff[grepl('Franklin', sff$County), ]

    Walla_Walla <- sff[grepl('Walla Walla', sff$County), ]
    Pend_Oreille <- sff[grepl('Pend Oreille', sff$County), ]

    Stevens <- sff[grepl('Stevens', sff$County), ]
    Spokane <- sff[grepl('Spokane', sff$County), ]

    Whitman <- sff[grepl('Whitman', sff$County), ]
    Garfield <- sff[grepl('Garfield', sff$County), ]

    Columbia <- sff[grepl('Columbia', sff$County), ]
    Asotin <- sff[grepl('Asotin', sff$County), ]
    Cowlitz <- sff[grepl('Cowlitz', sff$County), ]
  
  } else{
  	Adams <- sff[grepl('Adams', sff$county), ]
    Asotin <- sff[grepl('Asotin', sff$county), ]
    Benton <- sff[grepl('Benton', sff$county), ]
    Columbia <- sff[grepl('Columbia', sff$county), ]
    Cowlitz<- sff[grepl('Cowlitz', sff$county), ]
    Chelan <- sff[grepl('Chelan', sff$county), ]
    Douglas <- sff[grepl('Douglas', sff$county), ]
    Ferry <- sff[grepl('Ferry', sff$county), ]
    Franklin <- sff[grepl('Franklin', sff$county), ]
    Grant <- sff[grepl('Grant', sff$county), ]
    Garfield <- sff[grepl('Garfield', sff$county), ]
    Kittitas <- sff[grepl('Kittitas', sff$county), ]
    Klickitat <- sff[grepl('Klickitat', sff$county), ]
    Lincoln <- sff[grepl('Lincoln', sff$county), ]
    Okanogan <- sff[grepl('Okanogan', sff$county), ]
    Pend_Oreille <- sff[grepl('Pend Oreille', sff$county), ]
    Stevens <- sff[grepl('Stevens', sff$county), ]
    Spokane <- sff[grepl('Spokane', sff$county), ]
    Walla_Walla <- sff[grepl('Walla Walla', sff$county), ]
    Whitman <- sff[grepl('Whitman', sff$county), ]
    Yakima <- sff[grepl('Yakima', sff$county), ]
    
  }


  sff <- raster::bind(Adams, Asotin, Benton, Columbia, Cowlitz,
  	                  Chelan, Douglas, Ferry, Franklin, Grant, Garfield, Kittitas, Klickitat,
  	                  Lincoln, Okanogan, Pend_Oreille, Stevens, Spokane, Walla_Walla, Whitman,
  	                  Yakima)
  return(sff)

}

setnames(WSDA@data, "County", "county")
setnames(WSDA@data, "CropType", "CropTyp")

potato<-WSDA[grepl('Potato', WSDA$CropTyp), ]

corn_sweet<-WSDA[grepl('Corn, Sweet', WSDA$CropTyp), ]
corn_seed<-WSDA[grepl('Corn Seed', WSDA$CropTyp), ]
corn_field<-WSDA[grepl('Corn, Field', WSDA$CropTyp), ]
corn <- raster::bind(corn_sweet, corn_seed, corn_field)

writeOGR(obj = potato, 
         dsn = paste0(SF_dir, "/nitrogen_potato_irr_correctYr/"), 
         layer="nitrogen_potato_irr_correctYr", 
         driver="ESRI Shapefile")

writeOGR(obj = corn, 
         dsn = paste0(SF_dir, "/nitrogen_corn_irr_correctYr/"), 
         layer="nitrogen_corn_irr_correctYr", 
         driver="ESRI Shapefile")

corn_potato <- raster::bind(corn, potato)
writeOGR(obj = corn_potato, 
         dsn = paste0(SF_dir, "/nitrogen_corn_potato_irr_correctYr/"), 
         layer="nitrogen_corn_irr_correctYr", 
         driver="ESRI Shapefile")

corn_potato_data <- corn_potato@data
yr <- 2020
write.csv(corn_potato_data, 
          file = paste0(SF_dir, "corn_potato_data", yr, ".csv"), 
          row.names=FALSE)