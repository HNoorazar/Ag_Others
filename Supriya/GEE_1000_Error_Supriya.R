rm(list=ls())
library(data.table)
library(rgdal)
library(sp)
library(dplyr)
library(ggplot2)
library(maps)
library(purrr)
library(scales) # includes pretty_breaks

##############################################################################################################
#
#    Directory setup
#

SF_dir <- paste0("/Users/hn/Downloads/GEE_Error_when_features_have_more_than_1000000_vertices/")

GEE_1000_error <- readOGR(paste0(SF_dir, "AZ_2020.shp"),
                          layer = "AZ_2020", 
                          GDAL1_integer64_policy = TRUE)


grid_id_31 <- GEE_1000_error@data[31,]$grid_id
poly_31 <- GEE_1000_error[GEE_1000_error@data$grid_id == grid_id_31, ]
ggplot() +
geom_polygon(data = poly_31, aes(x=long, y=lat, group=group), fill = "grey47", color = "red")



writeOGR(obj = GEE_1000_error, 
         dsn = paste0(SF_dir, "/AZ_2020_R/"), 
         layer="AZ_2020_R", 
         driver="ESRI Shapefile")


# If you want to simplify:
GEE_1000_error_simple <- rmapshaper::ms_simplify(GEE_1000_error)



states <- map_data("state")
states_cluster <- subset(states, region %in% c("arizona"))

Error_map <- ggplot() +
             geom_polygon(data = states_cluster, 
                          aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
             geom_polygon(data = GEE_1000_error, 
                          aes(x=long, y=lat, group=group), fill = "grey47", color = "red") + 
             coord_sf(xlim = c(-1500000, -1000000), ylim = c(1100000, 1500000)) # <----- see if this works

plot_w <- 5
plot_h <- 5
ggsave(filename = paste0("Error_map.pdf"),
       plot = Error_map, 
       width = plot_w, height = plot_h, units = "in", limitsize = FALSE,
       dpi = 400, device = "pdf",
       path = SF_dir)


# If you want to simplify:
GEE_1000_error_simple <- rmapshaper::ms_simplify(GEE_1000_error)

Error_map_simple <- ggplot() +
                    geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
                    geom_polygon(data = GEE_1000_error_simple, aes(x=long, y=lat, group=group), fill = "grey47", color = "red") + 
                    coord_sf(xlim = c(-1500000, -1000000), ylim = c(1100000, 1500000)) # <----- see if this works

ggsave(filename = paste0("Error_map_simple.pdf"),
       plot = Error_map_simple, 
       width = plot_w, height = plot_h, units = "in", limitsize = FALSE,
       dpi = 400, device = "pdf",
       path = SF_dir)




###################################
###################################   CRS change
###################################
crs <- CRS("+proj=lcc 
           +lat_1=45.83333333333334 
           +lat_2=47.33333333333334 
           +lat_0=45.33333333333334 
           +lon_0=-120.5 +datum=WGS84")

GEE_1000_error_CRS <- spTransform(GEE_1000_error, 
                                  CRS("+proj=longlat +datum=WGS84"))

writeOGR(obj = GEE_1000_error_CRS, 
         dsn = paste0(SF_dir, "/AZ_2020_R_CRS/"), 
         layer="AZ_2020_R_CRS", 
         driver="ESRI Shapefile")


Error_map_CRS <- ggplot() +
                 geom_polygon(data = states_cluster, 
                              aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
                 geom_polygon(data = GEE_1000_error_CRS, 
                              aes(x=long, y=lat, group=group), fill = "grey47", color = "red")

plot_w <- 5
plot_h <- 5
ggsave(filename = paste0("Error_map_CRS.pdf"),
       plot = Error_map_CRS, 
       width = plot_w, height = plot_h, units = "in", limitsize = FALSE,
       dpi = 400, device = "pdf",
       path = SF_dir)

###################################
###################################
###################################
#
# If you want to simplify:
#
GEE_1000_error_CRS_simple <- rmapshaper::ms_simplify(GEE_1000_error_CRS)

writeOGR(obj = GEE_1000_error_CRS_simple, 
         dsn = paste0(SF_dir, "/AZ_2020_R_CRS_simple/"), 
         layer="AZ_2020_R_CRS_simple", 
         driver="ESRI Shapefile")


Error_map_CRS_simple <- ggplot() +
                        geom_polygon(data = states_cluster, 
                                     aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
                        geom_polygon(data = GEE_1000_error_CRS_simple, 
                                     aes(x=long, y=lat, group=group), fill = "grey47", color = "red")

ggsave(filename = paste0("Error_map_CRS_simple.pdf"),
       plot = Error_map_CRS_simple, 
       width = plot_w, height = plot_h, units = "in", limitsize = FALSE,
       dpi = 400, device = "pdf",
       path = SF_dir)




############################
#
#  Find and remove the bad polygon
#

#
#  When uploading the CRS version of the SF, GEE said: Error: 
#               Primary geometry of feature '31' has 1340524 vertices, above the limit of 1000000 vertices.
#
# So, I am removing the 31st polygon. 
#


grid_id_31 <- GEE_1000_error_CRS@data[31,]$grid_id
poly_31 <- GEE_1000_error_CRS[GEE_1000_error_CRS@data$grid_id == grid_id_31, ]
ggplot() +
geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
geom_polygon(data = poly_31, aes(x=long, y=lat, group=group), fill = "grey47", color = "red")



for (row_number in c(1:dim(GEE_1000_error_CRS@data)[1])){
  grid_id <- GEE_1000_error_CRS@data[row_number,]$grid_id
  poly <- GEE_1000_error_CRS[GEE_1000_error_CRS@data$grid_id == grid_id, ]
  a_polygon_plot <- ggplot() +
                    geom_polygon(data = states_cluster, 
                                 aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
                    geom_polygon(data = poly, aes(x=long, y=lat, group=group), fill = "grey47", color = "red")

  ggsave(filename = paste0("polygon_", grid_id, ".pdf"),
         plot = a_polygon_plot, 
         width = plot_w, height = plot_h, units = "in", limitsize = FALSE,
         dpi = 400, device = "pdf",
         path = SF_dir)

}


for (row_number in c(1:dim(GEE_1000_error_CRS@data)[1])){
  grid_id <- GEE_1000_error_CRS@data[a_row_number, ]$grid_id
  remove_31_poly <- GEE_1000_error_CRS[GEE_1000_error_CRS@data$grid_id != grid_id, ]
  writeOGR(obj = remove_31_poly, 
           dsn = paste0(SF_dir, "/remove_grid_id_", grid_id, "_polygon"), 
           layer= paste0("/remove_grid_id_", grid_id, "_polygon"), 
           driver="ESRI Shapefile")

}


