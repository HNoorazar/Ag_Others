library(rgdal)   # package for geospatial analysis

library(ncdf4)   # package for netcdf manipulation
library(raster)  # package for raster manipulation
library(data.table)
library(dplyr)


nc_data_6km_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/Tb_data_19_GHz_6.25km/"
dates_list <- list.dirs(nc_data_6km_dir, full.names = FALSE, recursive = FALSE)

# nc_file <- nc_open(paste0(nc_data_6km_dir, dates_list[1], '/NSIDC-0630-EASE2_N6.25km-F18_SSMIS-2013001-19V-E-SIR-CSU-v1.3.nc'))
# sink(paste0(nc_data_dir, "/", dates_list[1], '/NSIDC-0630-EASE2_N6.25km-F18_SSMIS-2013001-19V-E-SIR-CSU-v1.3.txt'))
# print(nc_file)
# sink()

raster_file = raster::raster(paste0(nc_data_6km_dir, dates_list[1], '/NSIDC-0630-EASE2_N6.25km-F18_SSMIS-2013001-19V-E-SIR-CSU-v1.3.nc'))

# raster_file_laea_wgs84    <- projectRaster(from=raster_file, crs='+proj=laea    +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs') # , NAflag=666
# raster_file_longlat_wgs84 <- projectRaster(from=raster_file, crs='+proj=longlat +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')

# Why dimentions of the followings are different?
# raster_points_dt = data.table(rasterToPoints(raster_file))
# raster_points_laea_wgs84   = data.table(rasterToPoints(raster_file_laea_wgs84))
# raster_points_latlong_wgs84= data.table(rasterToPoints(raster_file_longlat_wgs84))

raster_points_spatial <- rasterToPoints(raster_file, spatial=TRUE)
# r_pts_latlog      <- spTransform(raster_points_spatial, CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0" ))
r_pts_latlog_orig <- spTransform(raster_points_spatial, CRS("+proj=longlat +datum=WGS84 +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs"))


data_latlog_orig=data.table()
r_pts_longlat_orig_coord = data.table(r_pts_latlog_orig@coords)

data_latlog_orig$long <- r_pts_latlog_orig$x
data_latlog_orig$lat <- r_pts_latlog_orig$y
data_latlog_orig$T_B <- r_pts_latlog_orig@data$SIR.TB
data_latlog_orig$date <- dates_list[1]

data_latlog_orig <- data.table(data_latlog_orig)
data_latlog_orig <- data_latlog_orig %>% filter(long >= -125) %>% data.table()
data_latlog_orig <- data_latlog_orig %>% filter(long <= -109) %>% data.table()
data_latlog_orig <- data_latlog_orig %>% filter(lat >= 41) %>% data.table()
data_latlog_orig <- data_latlog_orig %>% filter(lat <= 53) %>% data.table()

# Do what we did above to all dates and save the
# result in one giant csv file. Do it for both 6-km and 3-km
rm(data_latlog_orig)

data_base <- "/Users/hn/Documents/01_research_data/Bhupi/snow/"
sub_dirs <- c("Tb_data_19_GHz_6.25km", "Tb_data_37GHz_3.125km")

for (a_subdir in sub_dirs){
  print (a_subdir)
  data_in <- paste0(data_base, "/", a_subdir, "/")
  dates_list <- list.dirs(data_in, full.names = FALSE, recursive = FALSE)
  all_TBs <- data.table()
  for (a_date in dates_list){
    curr_dir <- paste0(data_in, a_date, "/")
    file_list = list.files(path = curr_dir, pattern = ".nc") # we would have only one file in there! Yet, lets do a for-loop
    if (length(file_list)>1){
      print (paste0("There are more than one file in ", curr_dir))
    } else if (length(file_list)==0){
      print (paste0("*****  no file here: ", a_subdir, ", ", a_date))
    } else if (length(file_list)==1){
      print (paste0(a_subdir, ", ", a_date))
      raster_file = raster::raster(paste0(curr_dir, file_list[1]))
      raster_points_spatial <- rasterToPoints(raster_file, spatial=TRUE)
      r_pts_latlog_orig <- spTransform(raster_points_spatial, CRS("+proj=longlat +datum=WGS84 +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs"))

      data_latlog_orig <- data.table()
      r_pts_longlat_orig_coord <- data.table(r_pts_latlog_orig@coords)
      data_latlog_orig$long <- r_pts_latlog_orig$x
      data_latlog_orig$lat  <- r_pts_latlog_orig$y
      data_latlog_orig$T_B  <- r_pts_latlog_orig@data$SIR.TB
      data_latlog_orig$date <- a_date

      data_latlog_orig <- data.table(data_latlog_orig)
      data_latlog_orig <- data_latlog_orig %>% filter(long >= -125) %>% data.table()
      data_latlog_orig <- data_latlog_orig %>% filter(long <= -109) %>% data.table()
      data_latlog_orig <- data_latlog_orig %>% filter(lat >= 41) %>% data.table()
      data_latlog_orig <- data_latlog_orig %>% filter(lat <= 53) %>% data.table()

      all_TBs <- rbind(all_TBs, data_latlog_orig)
    }
  }
  write.csv(all_TBs,
            file = paste0(data_in, "all_", a_subdir, ".csv"),
            row.names=FALSE)
}

# Longitude min: -124.9022
# Longitude max: -109.7625
# Latitude min: 41.12783
# Latitude max: 52.88065

