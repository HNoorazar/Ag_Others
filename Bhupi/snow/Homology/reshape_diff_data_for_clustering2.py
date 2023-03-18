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
# This is a copy of ```reshape_diff_data_for_clustering```. In that notebook we skiped 2003 since in that year only 15 locations were present. Here we keep it, and clean(?) the data and save it in 2 different shapes. Some ML needs each row to be a datapoint (each column is a time step -a feature/dimension- and each row is a location) but I liked it the otherway for some reason!
#
# In this notebook each column will be a day, like DoY=1, ... 365, but could belong to different years. Exact date will be recorded in a column.

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

from pylab import imshow
import pickle
import h5py
import sys

# %%
snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EithyYearsClustering/"
diff_dir = snow_TS_dir_base+ "Brightness_temperature/Only_for_SNOTEL_grids/"

# %%
CSV_files = []
# Iterate directory
for file in os.listdir(diff_dir):
    # check only csv files
    if file.endswith('.csv'):
        CSV_files.append(file)

CSV_files = sorted(CSV_files)
print(CSV_files[0:2])
print(CSV_files[-2:])

# %%
# # Do this to figure out coordinate and use as column order later
# DF = pd.read_csv(diff_dir + CSV_files[0])
# DF.drop(labels=["x", "y"], axis="columns", inplace=True)
# coordinates = DF.lat_lon
# del(DF)
# column_order = list(coordinates) + ["month", "day", "year", "date"]

# %%
day_arr = list(np.repeat("day_", 365))
day_count = list(np.arange(1, 366))
col_names = [i + str(j) for i, j in zip(day_arr, day_count)]
col_names = col_names + ["lat_lon"]

# %% [markdown]
# ### Year 2003 has only 15 stations in it. Skip it!

# %%
all_stations = pd.DataFrame()
for a_csv in CSV_files:
#     if a_csv=="Year_2003.csv":
#         continue
    DF = pd.read_csv(diff_dir + a_csv)
    DF.drop(labels=["x", "y"], axis="columns", inplace=True)
    if DF.shape[1]==367:
        DF.drop(labels=DF.columns[-2], axis="columns", inplace=True)
    # new_col_names = [colName.replace(".", "_") for colName in list(DF.columns)]
    DF.columns = col_names
    
    year = int(a_csv.split("_")[1].split(".")[0])
    DF["year"]=year
    DF.reset_index(drop=True, inplace=True)
    all_stations = pd.concat([all_stations, DF])


# %%

# %%
output_name = "all_locs_all_years_eachDayAColumn.csv"
all_stations.to_csv(snow_TS_dir_base+ "Brightness_temperature/" + output_name, index=False)

# %%
all_stations.head(2)

# %%
