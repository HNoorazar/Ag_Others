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
# diff_dir = snow_TS_dir_base+ "Brightness_temperature/Only_for_SNOTEL_grids/"
SNOTEL_dir = snow_TS_dir_base + "SNOTEL_observations/"

# %%
CSV_files = []
# Iterate directory
for file in os.listdir(SNOTEL_dir):
    # check only csv files
    if file.endswith('.csv'):
        CSV_files.append(file)

CSV_files = sorted(CSV_files)
print(CSV_files)

# %%
SNOTEL_stations = pd.read_csv(SNOTEL_dir + CSV_files[0])
SNOTEL_stations.rename(columns={"Date": "date",
                                "Year": "year",
                                "Month": "month",
                                "Day": "day"}, inplace=True)
# new_cols = [a_col.lower() for a_col in SNOTEL_stations.columns[0:4]]
# SNOTEL_join_PMW_grids = pd.read_csv(SNOTEL_dir + CSV_files[1])
# print (SNOTEL_join_PMW_grids.shape)
# SNOTEL_join_PMW_grids.head(3)

# %%
station_names = list(SNOTEL_stations.columns[4:].values)
years = sorted(SNOTEL_stations.year.unique())

# %%
print (SNOTEL_stations.shape)
SNOTEL_stations.head(3)

# %%
day_arr = list(np.repeat("day_", 365))
day_count = list(np.arange(1, 366))
col_names = [i + str(j) for i, j in zip(day_arr, day_count)]
col_names = col_names + ["year", "station_name"]

# %%
len(SNOTEL_stations.year.unique())*len(SNOTEL_stations.year.unique())

# %%
# in_dir = snow_TS_dir_base + "Brightness_temperature/"
# file_Name = "all_locs_all_years_eachDayAColumn.pkl"
# all_stations_years = pd.read_pickle(in_dir+file_Name)
# all_stations_years = all_stations_years["all_locs_all_years_eachDayAColumn"]
# all_stations_years.head(2)

# count=0
# for ii in all_stations_years.lat_lon.unique():
#     if ii in SNOTEL_join_PMW_grids.PMW_lat_lon.unique():
#         count+=1
# count

# %%
# # Do this to figure out coordinate and use as column order later
# DF = pd.read_csv(diff_dir + CSV_files[0])
# DF.drop(labels=["x", "y"], axis="columns", inplace=True)
# coordinates = DF.lat_lon
# del(DF)
# column_order = list(coordinates) + ["month", "day", "year", "date"]

# %%

# %%
# %%time
repeated_years = years*len(station_names)
repeated_stations = station_names*len(years)

all_stations = pd.DataFrame(columns=col_names, index=range(len(years)*len(station_names)))

all_stations.year=repeated_years
all_stations.station_name=repeated_stations
all_stations.sort_values(by=["station_name", "year"], inplace=True)

all_stations.reset_index(drop=True, inplace=True)
print (all_stations.shape)

for a_station in station_names:
    curr_loc_data = SNOTEL_stations[["date", "year", "month", "day", a_station]]
    for a_year in years:
        curr_loc_year_data = curr_loc_data[curr_loc_data.year==a_year]
        curr_loc_year_data_values = curr_loc_year_data[a_station][0:365].values
        all_stations.loc[(all_stations.year==a_year) & 
                         (all_stations.station_name==a_station), "day_1":"day_365"]=curr_loc_year_data_values

all_stations.reset_index(drop=True, inplace=True)

# %%
all_stations.head(5)

# %%
all_locs_all_years_eachDayAColumn_dict={"all_locs_all_years_eachDayAColumn": all_stations,
                                        "jupyterNotebook_GeneratedThisdata": "reshape_SNOTEL_data_4_clustering"}

output_dir = SNOTEL_dir
os.makedirs(output_dir, exist_ok=True)

file_Name = "all_locs_all_years_eachDayAColumn_SNOTEL.pkl"
f = open(output_dir + file_Name, "wb") # create a binary pickle file 
pickle.dump(all_locs_all_years_eachDayAColumn_dict, f) # write the python object (dict) to pickle file
f.close() # close file
