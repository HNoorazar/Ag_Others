# .libPaths("/data/hydro/R_libs35")
# .libPaths()

source_path = "/home/h.noorazar/Sid/sidFabio/read_binary_core.R" # Kamiak


source(source_path)
options(digits=9)

detect_county_byLatLong <- function(latLong_datatable){
    counties <- map('county', fill=TRUE, col="transparent", plot=FALSE)
    IDs <- sapply(strsplit(counties$names, ":"), function(x) x[1])
    counties_sp <- map2SpatialPolygons(counties, IDs=IDs,
                     proj4string=CRS("+proj=longlat +datum=WGS84"))

    # Convert latLong_datatable to a SpatialPoints object 
    pointsSP <- SpatialPoints(latLong_datatable, 
                    proj4string=CRS("+proj=longlat +datum=WGS84"))

    # Use 'over' to get _indices_ of the Polygons object containing each point 
    indices <- over(pointsSP, counties_sp)

    # Return the county names of the Polygons object containing each point
    countyNames <- sapply(counties_sp@polygons, function(x) x@ID)
    countyNames[indices]
}

compute_GDD_nonLinear <- function(data_dir, file_name, data_type_, lower_cut, upper_cut, T_opt){
    #
    #
    # data_dir : is from Aeolus.
    # file_name : name of binary files on aeolus.
    # data_type_ : either "observed" or "future".
    #              This determines the number of variables in the binary files
    #              which helps to read the binary files correctly
    # lower_cut : Lower base temperature
    # upper_cut : Upper base temperature
    # T_opt : optimal temperature from the Claudio's non-linear equation.
    #

    #
    #  on Kamiak everything has 8 variales
    #
    # future data are all over the place. West are in Adams directory
    # non-west are elsewhere.  Hence this if-else statement.
    # right this second (Sept. 2022 we are doing observed and future (i.e. no modeled historical))

    no_vars_=8
    
    if (data_type_=="observed"){
        met_data <- read_binary(file_path=paste0(data_dir, file_name), 
                                data_type=data_type_, 
                                no_vars=no_vars_)
      } else{
        # future data are all over the place. West are in Adams directory
        # non-west are elsewhere. Hence this if-else statement.
        # right this second (Sept. 2022 we are doing observed and future (i.e. no modeled historical))
        met_data <- read_binary(file_path=paste0(file_name), 
                                data_type=data_type_, 
                                no_vars=no_vars_)
    }

    met_data <- data.table(met_data)

    # we do not need 1979.
    if (data_type_=="observed"){
      met_data <- met_data[met_data$year>=1980]
    }

    met_data$Tavg = (met_data$tmax+met_data$tmin)/2

    alpha = log(2)/log((upper_cut-lower_cut)/(T_opt-lower_cut))

    met_data$TAvg_Minus_lower_cut=met_data$Tavg-lower_cut

    met_data$numerator_p1=2*(met_data$TAvg_Minus_lower_cut * (T_opt-lower_cut)) ** alpha
    met_data$numerator_p2=met_data$TAvg_Minus_lower_cut**(2*alpha)
    met_data$numerator=met_data$numerator_p1 - met_data$numerator_p2

    denominator = (T_opt-lower_cut)**(2*alpha)

    met_data$WEDR = met_data$numerator/denominator
    
    # kill the days when Tavg is outside the lower and upper boundaries.
    met_data[Tavg < lower_cut, WEDR := 0]
    met_data[Tavg > upper_cut, WEDR := 0]
    met_data$WEDD = met_data$WEDR * (T_opt-lower_cut)

    met_data <- data.table(met_data)

    # Drop extra columns
    met_data[, c("TAvg_Minus_lower_cut", "numerator_p1", "numerator_p2", "numerator"):=NULL]

    # add day of year
    met_data$doy = 1
    met_data[, doy := cumsum(doy), by=list(year)]
    return (met_data)
}


compute_GDD_linear <- function(data_dir, file_name, data_type_, lower_cut, upper_cut){
    #
    #  on Kamiak everything has 8 variales
    #
    # future data are all over the place. West are in Adams directory
    # non-west are elsewhere.  Hence this if-else statement.
    # right this second (Sept. 2022 we are doing observed and future (i.e. no modeled historical))

    no_vars_=8
    
    if (data_type_=="observed"){
        met_data <- read_binary(file_path=paste0(data_dir, file_name), 
                                data_type=data_type_, 
                                no_vars=no_vars_)
      } else{
        # future data are all over the place. West are in Adams directory
        # non-west are elsewhere. Hence this if-else statement.
        # right this second (Sept. 2022 we are doing observed and future (i.e. no modeled historical))
        met_data <- read_binary(file_path=paste0(file_name), 
                                data_type=data_type_, 
                                no_vars=no_vars_)
    }
    # we do not need 1979.
    if (data_type_=="observed"){
      met_data <- met_data[met_data$year>=1980]
    }

    met_data$Tavg = (met_data$tmax+met_data$tmin)/2

   # if Tavg<lower_cut then daily_GDD = 0
   met_data[Tavg < lower_cut, daily_GDD := 0]

   # if Tavg>upper_cut then daily_GDD = upper_cut - lower_cut
   met_data[Tavg > upper_cut, daily_GDD := (upper_cut-lower_cut)]

   # if lower_cut<Tavg<upper_cut then daily_GDD = Tavg - lower_cut
   met_data[Tavg >= lower_cut & Tavg <= upper_cut, daily_GDD := (Tavg-lower_cut)]
   
   # met_data[, cum_GDD := cumsum(daily_GDD)] # , by=list(year)

   # add day of year
   met_data$doy = 1
   met_data[, doy := cumsum(doy), by=list(year)]

   return (met_data)
}
