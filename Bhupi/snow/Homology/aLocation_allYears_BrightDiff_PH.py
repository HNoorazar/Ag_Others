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
# This is created after the meeting w/ Ananth. 
# Use each location in different years as a set that we need to do clustering on. 
#
# In this notebook we collect data of a location across all years in one set. That is a given dataset for which we compute persistent diagram and save it to the disk.

# %%
import numpy as np
import pandas as pd

from datetime import date, datetime
import time

import random
from random import seed
from random import random

import os, os.path
import shutil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

from pylab import imshow
import pickle
import h5py
import sys

# %%
import shutup
shutup.please()

# %%
# # !pip3 install ripser
# pip install --upgrade numpy
# # !pip3 install tadasets
# # !pip3 install kmapper
import ripser
from ripser import Rips #, ripser

import persim
# from persim import plot_diagrams

import tadasets
import kmapper as km # Import the class

# %%
snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EithyYearsClustering/"
in_dir = snow_TS_dir_base + "Brightness_temperature/"

# %%
fileName = "all_locs_all_years_eachDayAColumn.csv"
all_stations_years = pd.read_csv(in_dir + fileName)
all_stations_years.head(2)

# %% [markdown]
# # Smooth

# %%
locations = all_stations_years["lat_lon"].unique()
years = all_stations_years["year"].unique()
print (len(locations))

# %%
# %%time
all_stations_years_smooth = all_stations_years.copy()

window_5 = 5
weights_5 = np.arange(1, window_5+1)

# weights_5 = np.arange(1, window_5-1)
# weights_5 = 1/weights_5
# weights_5 = 

for a_loc in locations:
    curr_loc = all_stations_years_smooth[all_stations_years_smooth.lat_lon==a_loc]
    years = curr_loc["year"].unique() # year 2003 does not have all locations!
    for a_year in years:
        a_signal = curr_loc.loc[curr_loc.year==a_year, "day_1":"day_365"]
        curr_idx = curr_loc.loc[curr_loc.year==a_year, "day_1":"day_365"].index[0]
        a_signal = pd.Series(a_signal.values[0])
        
        # moving average:
        # ma5=a_signal.rolling(window_size).mean().tolist()
        
        # weighted moving average. weights are not symmetric here.
        wma_5 = a_signal.rolling(window_5, center=False).apply(lambda a_signal: np.dot(a_signal, weights_5)/
                                                                   weights_5.sum(), raw=True)
        all_stations_years_smooth.loc[curr_idx, "day_1":"day_365"]=wma_5.values

all_stations_years_smooth.head(3)

# %% [markdown]
# We lost some data at the beginning due to rolling window. So, we replace them here:

# %%
end = window_5-1
NA_columns=list(all_stations_years_smooth.columns[0:end])
a_col = NA_columns[1]
for a_col in NA_columns:
    all_stations_years_smooth.loc[:, a_col] = all_stations_years_smooth.iloc[:, end]

all_stations_years_smooth.head(3)


# %%
def diagram_sizes(dgms):
    return ", ".join([f"|$H_{i}$|={len(d)}" for i, d in enumerate(dgms)])


# %%
a_loc = locations[0]
a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.lat_lon==a_loc]
# a_loc_specific_years = a_loc_data.year.unique()
# a_year = a_loc_specific_years[9]
# a_year_data = a_loc_data.loc[a_loc_data.year==a_year]

# %%
# ripser.ripser(all_locs_smooth_after_2004[["time_xAxis", "48.97191_-121.05145"]])['dgms']
a_dmg = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"])['dgms']
persim.plot_diagrams(a_dmg, show=False, title=f"rips output\n{diagram_sizes(a_dmg)}")

# %%
# output dir
output=in_dir + "aLocation_allYears_grouped/"
os.makedirs(output, exist_ok=True)

# %%
a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.lat_lon==a_loc]
ripser_output = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"], maxdim=2)
ripser_output["jupyterNotebook_GeneratedThisdata"] = "clustering_eachYearAdataPoint_BrightnessDiff"

file_Name = a_loc + "_" + str(len(a_loc_data.year.unique())) + "years_BrightDiff" + ".pkl"
f = open(output + file_Name, "wb") # create a binary pickle file 
pickle.dump(ripser_output, f) # write the python object (dict) to pickle file
f.close() # close file

filepath = output + file_Name
pickled_obj = pd.read_pickle(filepath)

# %%
for a_loc in locations:
    a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.lat_lon==a_loc]
    ripser_output = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"], maxdim=2)
    ripser_output["jupyterNotebook_GeneratedThisdata"] = "aLocation_allYears_BrightDiff_PH"

    file_Name = a_loc + "_" + str(len(a_loc_data.year.unique())) + "years_BrightDiff" + ".pkl"
    f = open(output + file_Name, "wb") # create a binary pickle file 
    pickle.dump(ripser_output, f) # write the python object (dict) to pickle file
    f.close() # close file
    a_dmg[1].shape

# %%
a_loc = locations[10]
a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.lat_lon==a_loc]
dgms = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms']
persim.plot_diagrams(dgms, show=True, lifetime=True, legend=True)

# %%
# persim.sliced_wasserstein(dgms[1], dgms[1])

# %%

# %%

# %%

# %%
