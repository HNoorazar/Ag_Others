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

# %% [markdown]
# In this notebook, I read the ```TB``` data of grids that are closes to the stations and subtract them to see what we get. In this notebook I am using grids that are closes to each station. In other words, resampling is not done here.

# %%
import pandas as pd
import numpy as np

import scipy
import scipy.signal
import os, os.path
import datetime
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sb

import sys
import io

from pylab import rcParams

import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

# %%
snow_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/"

# %%
stations_closest_grids_3km_6km = pd.read_csv(snow_dir_base+"stations_closest_grids_3km_6km.csv")
closest2stations_19GHz_6km = pd.read_csv(snow_dir_base + "closest2stations_Tb_data_19_GHz_6.25km.csv")
closest2stations_37GHz_3km = pd.read_csv(snow_dir_base + "closest2stations_Tb_data_37GHz_3.125km.csv")

# %%
stations_closest_grids_3km_6km.head(2)

# %%
stations_closest_grids_3km_6km.drop(['longitude', 'latitude'], axis=1, inplace=True)

# %%
closest2stations_19GHz_6km.head(2)

# %%
closest2stations_37GHz_3km.head(2)

# %%
print (stations_closest_grids_3km_6km.shape)
print (closest2stations_19GHz_6km.shape)
print (closest2stations_37GHz_3km.shape)

# %%
print (len(stations_closest_grids_3km_6km.closest_6km_Grid_coord.unique()))
print (len(stations_closest_grids_3km_6km.closest_3km_Grid_coord.unique()))
print (len(closest2stations_19GHz_6km.long_lat_6km.unique()))
print (len(closest2stations_37GHz_3km.long_lat_3km.unique()))

# %%
stations_closest_grids_3km_6km.head(2)

# %%
closest2stations_37GHz_3km.head(2)

# %%
closest2stations_19GHz_6km.head(2)

# %%
cols = ["long_lat_6km", "T_B_6km", "date_6km"]
cols2= ["Station_Name", "closest_6km_Grid_coord"]
closest2stations_19GHz_6km=pd.merge(closest2stations_19GHz_6km[cols],
                                    stations_closest_grids_3km_6km[cols2],
                                    right_on="closest_6km_Grid_coord", left_on="long_lat_6km", 
                                    how="left")
closest2stations_19GHz_6km.head(2)

# %%
print (len(closest2stations_19GHz_6km.closest_6km_Grid_coord))
sum(closest2stations_19GHz_6km.closest_6km_Grid_coord==closest2stations_19GHz_6km.long_lat_6km)

# %%
closest2stations_19GHz_6km.drop(['long_lat_6km'], axis=1, inplace=True)
closest2stations_19GHz_6km.head(2)

# %%
closest2stations_37GHz_3km.head(2)

# %%
cols = ["long_lat_3km", "T_B_3km", "date_3km"]
cols2= ["Station_Name", "closest_3km_Grid_coord"]
closest2stations_37GHz_3km=pd.merge(closest2stations_37GHz_3km[cols],
                                    stations_closest_grids_3km_6km[cols2],
                                    right_on="closest_3km_Grid_coord", left_on="long_lat_3km", 
                                    how="left")
closest2stations_37GHz_3km.head(2)

# %%
print (len(closest2stations_37GHz_3km.closest_3km_Grid_coord))
sum(closest2stations_37GHz_3km.closest_3km_Grid_coord==closest2stations_37GHz_3km.long_lat_3km)

# %%
closest2stations_37GHz_3km.drop(['long_lat_3km'], axis=1, inplace=True)
closest2stations_37GHz_3km.head(2)

# %%
closest2stations_19GHz_6km.head(2)

# %%
print (len(closest2stations_37GHz_3km.date_3km.unique()))
print (len(closest2stations_19GHz_6km.date_6km.unique()))

# %%
# sorted(closest2stations_37GHz_3km.date_3km.unique())
# sorted(closest2stations_19GHz_6km.date_6km.unique())

# %% [markdown]
# # Missing dates
# In the ```netcdf```  files Bhupi gave me, there was nothing in ```20130403``` for the 6-km. 
#
# Also, ```20130310``` is missing from 3-km grids for some girds!
#
# So, lets toss that date
# in 3km as well, so out subtractions are true

# %%
three_dates = closest2stations_37GHz_3km.date_3km.unique()
six_dates   = closest2stations_19GHz_6km.date_6km.unique()

print ([a for a in three_dates if not (a in list(six_dates))])
print ([a for a in six_dates if not (a in list(three_dates))])

# %%
closest2stations_37GHz_3km.shape

# %%
print (closest2stations_37GHz_3km.shape)
closest2stations_37GHz_3km = closest2stations_37GHz_3km[closest2stations_37GHz_3km.date_3km!=20130403]
print (closest2stations_37GHz_3km.shape)

print (closest2stations_19GHz_6km.shape)
closest2stations_19GHz_6km = closest2stations_19GHz_6km[closest2stations_19GHz_6km.date_6km!=20130310]
print (closest2stations_19GHz_6km.shape)

# %%
30351-30262

# %%
30886-30884

# %%
closest2stations_19GHz_6km.head(2)

# %%
closest2stations_37GHz_3km.head(2)

# %%
closest2stations_37GHz_3km[closest2stations_37GHz_3km.Station_Name=="Big Red Mountain"].shape

# %%
closest2stations_19GHz_6km[closest2stations_19GHz_6km.Station_Name=="Big Red Mountain"].shape

# %%
closest2stations_19GHz_6km[closest2stations_19GHz_6km.Station_Name=="Big Red Mountain"]

# %%
closest2stations_19GHz_6km.rename(columns={"date_6km": "date"}, inplace=True)
closest2stations_37GHz_3km.rename(columns={"date_3km": "date"}, inplace=True)

# %%
closest2stations_19GHz_6km.head(2)

# %%
closest2stations_37GHz_3km.head(2)

# %%
cols6km = ["Station_Name", "date", "T_B_6km"]
cols3km = ["Station_Name", "date", "T_B_3km"]

closest2stations_TBs_6km_3km = pd.merge(closest2stations_19GHz_6km[cols6km],
                                        closest2stations_37GHz_3km[cols3km],
                                        on=["Station_Name", "date"], how="left")

# %%
closest2stations_TBs_6km_3km.head(2)

# %%
closest2stations_TBs_6km_3km.tail(2)

# %%
# Difference between 19GHz and 37GHz
closest2stations_TBs_6km_3km["diff"] = closest2stations_TBs_6km_3km["T_B_6km"]-closest2stations_TBs_6km_3km["T_B_3km"]
closest2stations_TBs_6km_3km.head(2)

# %%
closest2stations_TBs_6km_3km["date"]=closest2stations_TBs_6km_3km["date"].astype(str)
# closest2stations_TBs_6km_3km["year"]=closest2stations_TBs_6km_3km['date'].str.slice(0, 4)
# closest2stations_TBs_6km_3km["month"]=closest2stations_TBs_6km_3km['date'].str.slice(4, 6)
# closest2stations_TBs_6km_3km["day"]=closest2stations_TBs_6km_3km['date'].str.slice(6, 8)

closest2stations_TBs_6km_3km["date"] = closest2stations_TBs_6km_3km['date'].str.slice(0, 4) + "-" + \
                                       closest2stations_TBs_6km_3km['date'].str.slice(4, 6) + "-" + \
                                       closest2stations_TBs_6km_3km['date'].str.slice(6, 8)

closest2stations_TBs_6km_3km["date"] = pd.to_datetime(closest2stations_TBs_6km_3km["date"])
closest2stations_TBs_6km_3km.tail(2)

# %%
closest2stations_TBs_6km_3km.sort_values(by=["Station_Name", "date"], inplace=True)
closest2stations_TBs_6km_3km.reset_index(drop=True, inplace=True)
closest2stations_TBs_6km_3km.head(2)

# %%
SNOTEL_dir = snow_dir_base + "00/SNOTEL_data/"
SNOTEL_Snow_depth_2013 = pd.read_csv(SNOTEL_dir + "SNOTEL_Snow_depth_2013.csv")
SNOTEL_Snow_depth_2013["Date"] = pd.to_datetime(SNOTEL_Snow_depth_2013["Date"])
SNOTEL_Snow_depth_2013.head(2)

# %%
stations=list(closest2stations_TBs_6km_3km.Station_Name.unique())

# %%
a_station = stations[2]
a_station = "Waterhole"
a_station = "Elk Butte"
a_station = "North Fork"
fig, ax2 = plt.subplots(1, 1, figsize=(15, 5), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(closest2stations_TBs_6km_3km.loc[closest2stations_TBs_6km_3km.Station_Name==a_station, "date"], 
         closest2stations_TBs_6km_3km.loc[closest2stations_TBs_6km_3km.Station_Name==a_station, "diff"], 
         linewidth = 2, ls = '-', label = 'PMW signal', c="k");

ax2.plot(SNOTEL_Snow_depth_2013["Date"], SNOTEL_Snow_depth_2013[a_station], 
         linewidth = 3, ls = '-.', label = 'SNOTEL', c="g");

# ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
# ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) #
# ax2.xaxis.set_major_locator(mdates.MonthLocator())
# ax2.xaxis.set_major_formatter(DateFormatter('%b'))
plt.ylim([-5, 15])

ax2.legend(loc="upper right");
ax2.grid(True);
# ax2.xaxis.set_major_locator(mdates.MonthLocator())
# ax2.xaxis.set_major_formatter(DateFormatter('%b'))


# %%

# %%

# %%
