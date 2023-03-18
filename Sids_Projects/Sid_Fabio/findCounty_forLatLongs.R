library(sp)
library(maps)
library(maptools)
library(data.table)

options(digit=9)
options(digits=9)


data_dir <- "/Users/hn/Documents/01_research_data/Sid/SidFabio/parameters/"
locations_fName <- "VIC_tomato_points.txt"

local_files <- read.delim(file = paste0(data_dir, locations_fName), 
                          header=FALSE, as.is=TRUE)

local_files <- as.vector(local_files$V1)
local_files_table <- data.table(local_files)

x <- sapply(local_files_table$local_files, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
local_files_table$lat = x[2, ]
local_files_table$long = x[3, ]

local_files_table$lat <- as.double(local_files_table$lat)
local_files_table$long <- as.double(local_files_table$long)

testPoints <- data.table(data.frame(x = c(-90, -120), y = c(44, 44)))

A <- local_files_table[, c("long", "lat")]
local_files_table$county <- detect_county_byLatLong(local_files_table[, c("long", "lat")])