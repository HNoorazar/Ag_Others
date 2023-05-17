

.libPaths("/data/hydro/R_libs35")
.libPaths()
library(ncdf4)   # package for netcdf manipulation
library(raster)  # package for raster manipulation
library(data.table)
library(rgdal) # package for geospatial analysis. This shit is not on Aeolus and I cannot install it.
library(dplyr)

options(digit=9)
options(digits=9)

start_time <- Sys.time()

data_base <- "/data/hydro/users/Hossein/Bhupi/snow/"
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

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)


# In install.packages("ncdf4", lib = "~/.local/lib/R3.5.1") :
#   installation of package ‘ncdf4’ had non-zero exit status
  