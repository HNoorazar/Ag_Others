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
    
all_stations.reset_index(drop=True, inplace=True)


# %%
all_stations.head(2)

# %%
# a_loc = "42.32438_-113.61324"
# b_loc = "42.69664_-118.61593"

# ii_data = all_stations.loc[all_stations.lat_lon==a_loc]
# jj_data = all_stations.loc[all_stations.lat_lon==b_loc]
# ii_data.equals(jj_data)

# %%

# %%
all_locs_all_years_eachDayAColumn_dict={"all_locs_all_years_eachDayAColumn": all_stations,
                                        "jupyterNotebook_GeneratedThisdata": "reshape_diff_data_for_clustering"}

output_dir = snow_TS_dir_base+ "Brightness_temperature/"
os.makedirs(output_dir, exist_ok=True)

file_Name = "all_locs_all_years_eachDayAColumn.pkl"
f = open(output_dir + file_Name, "wb") # create a binary pickle file 
pickle.dump(all_locs_all_years_eachDayAColumn_dict, f) # write the python object (dict) to pickle file
f.close() # close file

# %%
# output_name = "all_locs_all_years_eachDayAColumn.csv"
# all_stations.to_csv(snow_TS_dir_base+ "Brightness_temperature/" + output_name, index=False)

# %% [markdown]
# # Skip 2003 since there are only 15 locations in that year!

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
# Do this to figure out coordinate and use as column order later
DF = pd.read_csv(diff_dir + CSV_files[0])
DF.drop(labels=["x", "y"], axis="columns", inplace=True)
coordinates = DF.lat_lon
del(DF)
column_order = list(coordinates) + ["month", "day", "year", "date"]

# %%
### Year 2003 has only 15 stations in it. Skip it!

all_stations = pd.DataFrame()

for a_csv in CSV_files:
    if a_csv=="Year_2003.csv":
        continue
    DF = pd.read_csv(diff_dir + a_csv)
    DF.drop(labels=["x", "y"], axis="columns", inplace=True)
    new_col_names = [colName.replace(".", "_") for colName in list(DF.columns)]
    DF.columns = new_col_names

    # Reshape
    DF = DF.T
    DF.columns=list(DF.loc["lat_lon"])
    DF.drop(labels=["lat_lon"], axis="index", inplace=True)

    DF.reset_index(inplace=True)
    DF.rename(columns={"index": "alph_date"}, inplace=True)
    A = DF.alph_date.str.split(pat="_", expand=True)
    new_col_names = ["month", "day", "year"]
    A.columns = new_col_names

    A['month'] = pd.to_datetime(A['month'], format='%b').dt.month
    A.day = A.day.astype(int)
    A.year = A.year.astype(int)

    DF["month"]=A["month"]
    DF["day"]=A["day"]
    DF["year"]=A["year"]
    DF['date'] = pd.to_datetime(dict(year=DF.year, month=DF.month, day=DF.day))
    DF = DF[column_order]
    all_stations = pd.concat([all_stations, DF])
    
all_stations.reset_index(drop=True, inplace=True)

# %%
all_locs_all_years_but_2003_dict={"all_locs_all_years_but_2003": all_stations,
                                  "jupyterNotebook_GeneratedThisdata": "reshape_diff_data_for_clustering"}

output_dir = snow_TS_dir_base+ "Brightness_temperature/"
os.makedirs(output_dir, exist_ok=True)

file_Name = "all_locs_all_years_but_2003.pkl"
f = open(output_dir + file_Name, "wb") # create a binary pickle file 
pickle.dump(all_locs_all_years_eachDayAColumn_dict, f) # write the python object (dict) to pickle file
f.close() # close file

# all_stations.to_csv(snow_TS_dir_base+ "Brightness_temperature/" + "all_locs_all_years_but_2003.csv", index=False)

# %%
