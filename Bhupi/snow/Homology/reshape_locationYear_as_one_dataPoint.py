# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd


# %%
snow_TS_dir_base = "/Users/hn/Documents/data/EithyYearsClustering/"
snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EithyYearsClustering/"
diff_dir = snow_TS_dir_base + "Brightness_temperature/Only_for_SNOTEL_grids/"

# %%
all_locs_all_years_but_2003 = pd.read_csv(snow_TS_dir_base + \
                                          "Brightness_temperature/" + \
                                          "all_locs_all_years_but_2003.csv")
all_locs_all_years_but_2003.date = pd.to_datetime(all_locs_all_years_but_2003.date)
all_locs_all_years_but_2003.head(2)

# %%
a_loc = "42.32438_-113.61324"
b_loc = "42.69664_-118.61593"

ii_data = all_locs_all_years_but_2003[a_loc]
jj_data = all_locs_all_years_but_2003[b_loc]

ii_data==jj_data

# %%

# %%
all_locs_after_2004 = all_locs_all_years_but_2003[all_locs_all_years_but_2003.year>=2004].copy()
all_locs_after_2004.reset_index(drop=True, inplace=True)

# %%
locations = list(all_locs_after_2004.columns)

bad_cols = ["month", "day", "year", "date"]
for a_bad in bad_cols:
    locations.remove(a_bad)

print (f"{len(locations)=}")

# Check if our data is daily
print (len(all_locs_after_2004))
print ((all_locs_after_2004.year.max()-all_locs_after_2004.year.min()+1)*365)

# %% [markdown]
# ### Smooth window size 5

# %%
all_locs_smooth_after_2004 = all_locs_after_2004.copy()

window_5 = 5
weights_5 = np.arange(1, window_5+1)

for a_loc in locations:
    a_signal = all_locs_smooth_after_2004[a_loc]
    wma_5 = a_signal.rolling(window_5).apply(lambda a_signal: np.dot(a_signal, weights_5)/weights_5.sum(), raw=True)
    all_locs_smooth_after_2004[a_loc] = wma_5

all_locs_smooth_after_2004.head(10)

# %%
end = window_5-1
all_locs_smooth_after_2004.iloc[0:end, 0:len(locations)]=all_locs_smooth_after_2004.iloc[end, 0:len(locations)]

# all_locs_smooth_after_2004 = all_locs_smooth_after_2004.assign(time_xAxis=range(len(all_locs_smooth_after_2004)))
all_locs_smooth_after_2004.head(2)

# %%
##
##  round to 3 digits
##
all_locs_smooth_after_2004 = all_locs_smooth_after_2004.round(3)

# %%
a_loc = "42.32438_-113.61324"
b_loc = "42.69664_-118.61593"

ii_data = all_locs_smooth_after_2004[a_loc]
jj_data = all_locs_smooth_after_2004[b_loc]
ii_data==jj_data

# %%
years = all_locs_smooth_after_2004.year.unique()

# %%
all_locs_smooth_after_2004

# %%
day_arr = list(np.repeat("day_", 365))
day_count = list(np.arange(1, 366))
col_names = [i + str(j) for i, j in zip(day_arr, day_count)]

# row_names=[]
# for a_location in locations:
#     for a_date in all_locs_smooth_after_2004.date.unique():
#         ts = pd.to_datetime(str(a_date)) 
#         string_d = ts.strftime('%Y_%m_%d')
#         row_names.extend([a_location + "_" + string_d])

row_names=[]
for a_location in locations:
    row_names.extend([a_location + "_" + str(a_year) for a_year in all_locs_smooth_after_2004.year.unique()])
        
smoothed_yearLocSingleDataPoint = pd.DataFrame(columns=col_names, index=range(len(locations)*len(years)))
smoothed_yearLocSingleDataPoint["loc_year"] = row_names
print (smoothed_yearLocSingleDataPoint.shape)
smoothed_yearLocSingleDataPoint.head(2)

# %%
# %%time
col_a = "day_1"
col_z = "day_365"
for a_location in locations:
    for a_year in all_locs_smooth_after_2004.year.unique():
        loc_year = a_location+ "_"+ str(a_year)
        curr_TS = list(all_locs_smooth_after_2004.loc[all_locs_smooth_after_2004.year==a_year, a_location])[0:365]
        smoothed_yearLocSingleDataPoint.loc[smoothed_yearLocSingleDataPoint.loc_year==loc_year, col_a:col_z]=curr_TS
        
smoothed_yearLocSingleDataPoint.head(2)

# %%
a_loc = "42.32438_-113.61324"
b_loc = "42.69664_-118.61593"

ii_data = smoothed_yearLocSingleDataPoint[a_loc]
jj_data = smoothed_yearLocSingleDataPoint[b_loc]
ii_data==jj_data

# %%
all_stations.to_csv(snow_TS_dir_base+ "Brightness_temperature/" + "all_locs_all_years_but_2003.csv", index=False)

# %%

# %%

# %%

# %%

# %%

# %%

# %%
