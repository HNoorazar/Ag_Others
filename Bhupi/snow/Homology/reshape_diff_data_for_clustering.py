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
# Do this to figure out coordinate and use as column order later
DF = pd.read_csv(diff_dir + CSV_files[0])
DF.drop(labels=["x", "y"], axis="columns", inplace=True)
coordinates = DF.lat_lon
del(DF)
column_order = list(coordinates) + ["month", "day", "year", "date"]

# %% [markdown]
# ### Year 2003 has only 15 stations in it. Skip it!

# %%
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


# %%
all_stations.to_csv(snow_TS_dir_base+ "Brightness_temperature/" + "all_locs_all_years_but_2003.csv", index=False)
