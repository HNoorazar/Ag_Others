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
# Before, I found all the ```TB```s for 6-km and 3-km grids within bounding box of Columbia River Basin.
#
# I also found which coordinates among those grids are closest to the stations. 
#
# In this notebook I subset the ```TB``` data so that I have only the ```TB``` data of those closest grids. This makes
# the files smaller in size and gets rid of extra grids.

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
six_km_data_dir   = snow_dir_base + "Tb_data_19_GHz_6.25km/"
three_km_data_dir = snow_dir_base + "Tb_data_37GHz_3.125km/"

# %%
stations_closest_grids_3km_6km = pd.read_csv(snow_dir_base+"stations_closest_grids_3km_6km.csv")
all_Tb_data_19GHz_6km = pd.read_csv(six_km_data_dir   + "all_Tb_data_19_GHz_6.25km.csv")
all_Tb_data_37GHz_3km = pd.read_csv(three_km_data_dir + "all_Tb_data_37GHz_3.125km.csv")

# %% [markdown]
# ### Subset only closest grids.

# %%
print (stations_closest_grids_3km_6km.shape)
stations_closest_grids_3km_6km.head(2)

# %%
all_Tb_data_19GHz_6km.head(2)

# %%
all_Tb_data_37GHz_3km.head(2)

# %%

# %%
all_Tb_data_19GHz_6km["long_lat_6km"] = round(all_Tb_data_19GHz_6km.long, 5).astype(str) + "_" + \
                                        round(all_Tb_data_19GHz_6km.lat, 5).astype(str)
    
all_Tb_data_37GHz_3km["long_lat_3km"] = round(all_Tb_data_37GHz_3km.long, 5).astype(str) + "_" + \
                                        round(all_Tb_data_37GHz_3km.lat, 5).astype(str)

# %%
all_Tb_data_37GHz_3km.head(2)

# %%
close_6km = list(stations_closest_grids_3km_6km.closest_6km_Grid_coord)
all_Tb_data_19GHz_6km = all_Tb_data_19GHz_6km[all_Tb_data_19GHz_6km.long_lat_6km.isin(close_6km)]
len(all_Tb_data_19GHz_6km.long_lat_6km.unique())

# %%
close_3km = list(stations_closest_grids_3km_6km.closest_3km_Grid_coord)
all_Tb_data_37GHz_3km = all_Tb_data_37GHz_3km[all_Tb_data_37GHz_3km.long_lat_3km.isin(close_3km)]
len(all_Tb_data_37GHz_3km.long_lat_3km.unique())

# %%
all_Tb_data_19GHz_6km.rename(columns={"T_B": "T_B_6km"}, inplace=True)
all_Tb_data_37GHz_3km.rename(columns={"T_B": "T_B_3km"}, inplace=True)

all_Tb_data_19GHz_6km.rename(columns={"date": "date_6km"}, inplace=True)
all_Tb_data_37GHz_3km.rename(columns={"date": "date_3km"}, inplace=True)

# %%
out_name = snow_dir_base + "closest2stations_Tb_data_19_GHz_6.25km.csv"
all_Tb_data_19GHz_6km.to_csv(out_name, index = False)

# %%
out_name = snow_dir_base + "closest2stations_Tb_data_37GHz_3.125km.csv"
all_Tb_data_37GHz_3km.to_csv(out_name, index = False)

# %%
