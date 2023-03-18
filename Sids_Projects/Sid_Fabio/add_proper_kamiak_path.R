

# add proper path:

local_files[!(lat>=32 & lat<=53 & long>=-125 & long<=-109)]$path<-paste0("/data/project/agaid/rajagopalan_agroecosystems/", 
                                                                         "commondata/meteorologicaldata/gridded/GCMs/US_Conus/"

local_files[lat>=32 & lat<=53 & long>=-125 & long<=-109]$path <- "/data/adam/data/metdata/VIC_ext_maca_v2_binary_westUSA/"