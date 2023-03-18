
.libPaths("/data/hydro/R_libs35")
.libPaths()



source_path="/Users/hn/Documents/00_GitHub/Ag/read_binary_core/read_binary_core.R"
source_path = "/home/hnoorazar/reading_binary/read_binary_core.R"
source(source_path)


options(digits=9)
options(digits=9)



compute_GDD <- function(data_dir, file_name, observed_or_future, lower_cut=10, upper_cut=30, maturity_cumGDD=1214){


    if (observed_or_future=="observed"){
        no_vars_=8
      } else if (observed_or_future=="future"){
        no_vars_=4
    }
    met_data <- read_binary(file_path=paste0(data_dir, file_name), hist=observed_or_future, no_vars=no_vars_)

    # we do not need 1979.
    met_data <- met_data[met_data$year>=1980]

    met_data$Tavg = (met_data$tmax+met_data$tmin)/2

   # if Tavg<lower_cut then daily_GDD = 0
   met_data[Tavg < lower_cut, daily_GDD := 0]

   # if Tavg>upper_cut then daily_GDD = upper_cut - lower_cut
   met_data[Tavg > upper_cut, daily_GDD := (upper_cut-lower_cut)]

   # if lower_cut<Tavg<upper_cut then daily_GDD = Tavg - lower_cut
   met_data[Tavg >= lower_cut & Tavg <= upper_cut, daily_GDD := (Tavg-lower_cut)]
   
   met_data[, cum_GDD := cumsum(daily_GDD)] # , by=list(year)

   # add day of year
   met_data$doy = 1
   met_data[, doy := cumsum(doy), by=list(year)]

   return (met_data)
}



