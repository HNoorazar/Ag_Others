suitability <- function(file_path){

  print(file_path)

  path_split <- str_split(file_path, pattern = "/", simplify = TRUE)

  name_split <- str_split(path_split[,3], pattern = "_", simplify = TRUE)

  lat = as.numeric(name_split[,2])
  lon = as.numeric(name_split[,3])

  df <-
    read_gridmet(paste0("/data/project/agaid/rajagopalan_agroecosystems/commondata/", 
                         "meteorologicaldata/gridded/gridMET/gridmet/historical/", file_path), begin = 1990, end = 2020) %>%
    mutate(year = year(date), doy = yday(date), month = month(date)) 
  
  df %>%
    group_by(year, date, month, tmax, tmin, precip,SRAD,Rmax,Rmin) %>%
    mutate(tavg = (tmax + tmin)/2)%>%
    group_by(year, date, month) %>%
    
    ## find season length (when GDD reaches 1200) for a window moving every 15 days
    ## We will have several windows and then based on literature we select the optimal window
    ## We come up with a range
    
   
   


    ## This is second part
    ## for the window selected from above step we find the following variables
    
    summarise(cold_day = sum(tmin < 0),
	            sub_optimal_day = sum(tavg >= 7 & tavg <= 18),
	            optimal_day = sum(tavg > 18 & tavg <= 21),
              super_optimal_day = sum(tavg > 21 & tavg <= 30),
	            heatstress_30 = sum(tmax > 31),
              heatstress_40 = sum(tmax > 40),
              Rmax = sum(Rmax),
              Srad = sum (SRAD))  %>%
    mutate(lat = lat,
           lon = lon,
           model = path_split[, 1],
           climate_proj = path_split[, 2])
}