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
# ### Nov. 7.
#
# First attempt to smoothing snow stuff.

# %%
# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from math import factorial
import scipy
import scipy.signal
import os, os.path

# from pprint import pprint
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sb

import sys
import io

from pylab import rcParams
# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

from numpy.fft import rfft, irfft, rfftfreq, ifft
from scipy import fft
import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

# %%
snow_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/00/"
PMW_diff_dir = snow_dir + "PMW_difference_data/"
SNOTEL_dir = snow_dir + "SNOTEL_data/"

# %%
# PMW_badShape = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013.csv")
# PMW_badShape.rename(columns={"x": "longitude", "y":"latitude"}, inplace=True)
# PMW_badShape.to_csv(PMW_diff_dir + "PMW_difference_data_2013.csv", index=False)
# PMW_badShape.head(2)

# %%
SNOTEL_Snow_depth_2013 = pd.read_csv(SNOTEL_dir + "SNOTEL_Snow_depth_2013.csv")
PMW_difference_data_2013 = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013.csv")

print (SNOTEL_Snow_depth_2013.shape)
print (PMW_difference_data_2013.shape)

# %%
new_col_names = [file_.replace(".", "_") for file_ in list(PMW_difference_data_2013.columns)]
PMW_difference_data_2013.columns = new_col_names
PMW_difference_data_2013.head(2)

# %%
# # Subset for GEE
# PMW_subset = PMW_difference_data_2013[["longitude", "latitude", "lat_lon", "Station_Name"]]
# PMW_subset.to_csv(PMW_diff_dir + "PMW_subset_GEE.csv", index=False)

# %%

# %%
sorted(SNOTEL_Snow_depth_2013.columns[2:].unique())==sorted(PMW_difference_data_2013.Station_Name.unique())

# %%
SNOTEL_Snow_depth_2013.head(2)

# %%
station_names = list(SNOTEL_Snow_depth_2013.columns[2:])
station_names[0:4]

# %%
no_snow_first_DoY = pd.DataFrame(index=range(0, len(station_names)), 
                                 columns = ["Station_Name", "no_snow_first_DoY"])
no_snow_first_DoY.Station_Name = station_names
no_snow_first_DoY.head(2)

# %%
# %%time
window_size = 28
station_count = 0
for a_station in station_names:
    curr_station_data = SNOTEL_Snow_depth_2013[a_station]
    for row_idx in np.arange(len(curr_station_data)-window_size):
        moving_window=curr_station_data[row_idx:row_idx+window_size]
        is_there_any_snow = sum(moving_window)
        if is_there_any_snow==0:
            date_=SNOTEL_Snow_depth_2013.loc[row_idx, "Date"]
            no_snow_first_DoY.loc[no_snow_first_DoY.Station_Name==a_station, "no_snow_first_DoY"]=date_
            break

# %%
427/264

# %%
# %%time
SNOTEL_Snow_depth_2013.rolling(28).sum()

# %%
((427/89)*20000*(2020-1982+1))/1000/60

# %% [markdown]
# #### Shuffle the stations randomly
# to use for convolution traininig if needed

# %%
station_names = list(SNOTEL_Snow_depth_2013.columns[2:])
print (station_names[:4])

import random
random.seed(0)
random.shuffle(station_names)
station_names[:4]

train_size = int(0.8*len(station_names))

train_stations = station_names[:train_size]
test_stations = station_names[train_size:]

no_snow_first_DoY["set_category"] = "none"
no_snow_first_DoY.loc[no_snow_first_DoY.Station_Name.isin(train_stations), "set_category"]="train"
no_snow_first_DoY.loc[no_snow_first_DoY.Station_Name.isin(test_stations), "set_category"]="test"
no_snow_first_DoY.head(2)

# %%
no_snow_first_DoY.to_csv(snow_dir + "DayRule" + str(window_size) + "_no_snow_first_DoY.csv", index=False)

# %%
PMW_difference_data_2013.to_csv(PMW_diff_dir + "PMW_difference_data_2013.csv", index=False)

# %%
PMW_difference_data_2013.head(2)

# %% [markdown]
# ## Reshape PMW
# to have similar format to SNOTEL and easy manipulation

# %%
PMW_difference_data_2013 = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013.csv")

PMW_difference_data_2013.drop(labels=["x", "y", "lat_lon"], axis="columns", inplace=True)
new_col_names = [colName.replace(".", "_") for colName in list(PMW_difference_data_2013.columns)]
PMW_difference_data_2013.columns = new_col_names
PMW_difference_data_2013.head(2)

# Reshape
PMW_difference_data_2013 = PMW_difference_data_2013.T
PMW_difference_data_2013.columns=list(PMW_difference_data_2013.loc["Station_Name"])
PMW_difference_data_2013.drop(labels=["Station_Name"], axis="index", inplace=True)

PMW_difference_data_2013.reset_index(inplace=True)
PMW_difference_data_2013.rename(columns={"index": "alph_date"}, inplace=True)
PMW_difference_data_2013["Date"] = SNOTEL_Snow_depth_2013["Date"]
new_col_order = list(PMW_difference_data_2013.columns[-1:])+list(PMW_difference_data_2013.columns[:-1])
PMW_difference_data_2013=PMW_difference_data_2013[new_col_order]

PMW_difference_data_2013.to_csv(PMW_diff_dir + "PMW_difference_data_2013_goodShape.csv", index=False)

# %%
PMW_difference_data_2013 = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013.csv")

PMW_difference_data_2013.drop(labels=["x", "y", "lat_lon"], axis="columns", inplace=True)
new_col_names = [colName.replace(".", "_") for colName in list(PMW_difference_data_2013.columns)]
PMW_difference_data_2013.columns = new_col_names

PMW_goodShape = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013_goodShape.csv")


station_names = list(SNOTEL_Snow_depth_2013.columns[2:])

# %%
a_station=station_names[88]
sum(PMW_goodShape[a_station]-\
    PMW_difference_data_2013[PMW_difference_data_2013.Station_Name==a_station].values[0][:-1])


# %%
# SNOTEL_Snow_depth_2013[SNOTEL_Snow_depth_2013.Date >= "2013-03-08", a_station]

# %%

# %%
# df = SNOTEL_Snow_depth_2013.copy()
# df['Date'] = pd.to_datetime(df['Date'])
# start_idx = df[df.Date == "2013-03-08"].index[0]-1
# end_idx   = start_idx+27
# df.loc[start_idx:end_idx, a_station]

# %%
