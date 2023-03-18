
# Find closest coordinate to the stations


###############################################
###
###    Load Libraries
###
###############################################

library(data.table)
library(geosphere) # compute distances
library(dplyr)

###############################################
###
###    Directories
###
###############################################

six_kmGrids_dir <- "/Users/hn/Documents/01_research_data/Bhupi/snow/Tb_data_19_GHz_6.25km/"
three_kmGrids_dir <- "/Users/hn/Documents/01_research_data/Bhupi/snow/Tb_data_37GHz_3.125km/"
stationName_dir <- "/Users/hn/Documents/01_research_data/Bhupi/snow/00/PMW_difference_data/"

###############################################
###
###    Read Station coordinates
###
###############################################

stationNameCoordinate <- data.table(read.csv(paste0(stationName_dir, "PMW_stationNameCoordinate.csv")))

###############################################
###
### Find closest grid in 6km grids and add it 
### to the stationNameCoordinate
###
###############################################


###############################################
###
###    Read 6km Satellite Grids' Coordinates
###
###############################################

six_kmGrids_coords <- data.table(read.csv(paste0(six_kmGrids_dir, "all_Tb_data_19_GHz_6.25km.csv")))

# subset lat longs
# six_kmGrids_coords <- six_kmGrids_coords[, c("long", "lat")]

# We want to keep unique ones!
six_kmGrids_coords$long <- round(six_kmGrids_coords$long, 5)
six_kmGrids_coords$lat <- round(six_kmGrids_coords$lat, 5)
six_kmGrids_coords$longLat = paste0(six_kmGrids_coords$long, "_", six_kmGrids_coords$lat)

unique_coords_6km <- data.table(unique(six_kmGrids_coords$longLat))
rm(six_kmGrids_coords)

x <- sapply(unique_coords_6km$V1, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
long = x[1, ]
lat = x[2, ]

unique_coords_6km <- data.table()
unique_coords_6km$longitude = long
unique_coords_6km$latitude = lat

unique_coords_6km$longitude <- as.numeric(unique_coords_6km$longitude)
unique_coords_6km$latitude <- as.numeric(unique_coords_6km$latitude)

closest_6km_grids = distm(stationNameCoordinate[, c("longitude", "latitude")], 
                          unique_coords_6km,
                          fun=distGeo)


stationNameCoordinate$closest_6km_Grid_coord <- "0"

for (a_row in c(1:nrow(stationNameCoordinate))){
  min_idx = which.min(closest_6km_grids[a_row, ])
  stationNameCoordinate[a_row, "closest_6km_Grid_coord"] <- paste0(unique_coords_6km[min_idx, longitude], "_", 
                                                                   unique_coords_6km[min_idx, latitude])
}

###############################################
###
### Find closest grid in 3km grids and add it 
### to the stationNameCoordinate
###
###############################################


###############################################
###
###    Read 3km Satellite Grids' Coordinates
###
###############################################

three_kmGrids_coords <- data.table(read.csv(paste0(three_kmGrids_dir, "all_Tb_data_37GHz_3.125km.csv")))

# We want to keep unique ones!
three_kmGrids_coords$long <- round(three_kmGrids_coords$long, 5)
three_kmGrids_coords$lat <- round(three_kmGrids_coords$lat, 5)
three_kmGrids_coords$longLat = paste0(three_kmGrids_coords$long, "_", three_kmGrids_coords$lat)

unique_coords_3km <- data.table(unique(three_kmGrids_coords$longLat))
rm(three_kmGrids_coords)

x <- sapply(unique_coords_3km$V1, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
long = x[1, ]
lat = x[2, ]

unique_coords_3km <- data.table()
unique_coords_3km$longitude = long
unique_coords_3km$latitude = lat

unique_coords_3km$longitude <- as.numeric(unique_coords_3km$longitude)
unique_coords_3km$latitude  <- as.numeric(unique_coords_3km$latitude)

closest_3km_grids = distm(stationNameCoordinate[, c("longitude", "latitude")], 
                          unique_coords_3km,
                          fun=distGeo)


stationNameCoordinate$closest_3km_Grid_coord <- "0"

for (a_row in c(1:nrow(stationNameCoordinate))){
  min_idx = which.min(closest_3km_grids[a_row, ])
  stationNameCoordinate[a_row, "closest_3km_Grid_coord"] <- paste0(unique_coords_3km[min_idx, longitude], "_", 
                                                                   unique_coords_3km[min_idx, latitude])
}

out_dir <- "/Users/hn/Documents/01_research_data/Bhupi/snow/"
write.csv(stationNameCoordinate,
          file = paste0(out_dir, "stations_closest_grids_3km_6km.csv"),
          row.names=FALSE)




