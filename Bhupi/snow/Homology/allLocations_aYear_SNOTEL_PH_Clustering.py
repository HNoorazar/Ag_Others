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
# In this notebook we collect data of a year across all locations in one set. 
# That is a given dataset for which we compute persistent diagram and save it to the disk.

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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

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
SNOTEL_dir = snow_TS_dir_base + "SNOTEL_observations/"

# %%
SNOTEL_join_PMW_grids = pd.read_csv(SNOTEL_dir + "SNOTEL_join_PMW_grids.csv")
SNOTEL_join_PMW_grids.head(2)

# %%
file_Name = "all_locs_all_years_eachDayAColumn_SNOTEL.pkl"
all_stations_years = pd.read_pickle(SNOTEL_dir+file_Name)
all_stations_years = all_stations_years["all_locs_all_years_eachDayAColumn"]
all_stations_years.head(2)

# %% [markdown]
# # Smoothen

# %%
locations = all_stations_years["station_name"].unique()
locations=sorted(locations)
years = sorted(all_stations_years["year"].unique())
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
    curr_loc = all_stations_years_smooth[all_stations_years_smooth.station_name==a_loc]
    station_years = curr_loc["year"].unique() # year 2003 does not have all locations!
    for a_year in station_years:
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
data_2003 = all_stations_years_smooth[all_stations_years_smooth.year==2003].copy()
len(data_2003.station_name.unique())

# %%
years=list(all_stations_years_smooth.year.unique())

# %%

# %%
a_year = years[0]
a_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year]                           
# a_loc_specific_years = a_loc_data.year.unique()
# a_year = a_loc_specific_years[9]
# a_year_data = a_loc_data.loc[a_loc_data.year==a_year]
print (a_year_data.shape)
print (f"{len(a_year_data.station_name.unique())=}")

# ripser.ripser(all_locs_smooth_after_2004[["time_xAxis", "48.97191_-121.05145"]])['dgms']
a_dmg = ripser.ripser(a_year_data.loc[:, "day_1":"day_365"])['dgms']
persim.plot_diagrams(a_dmg, show=False, title=f"{a_year}, rips output\n{diagram_sizes(a_dmg)}")

# %%

# %%
SNOTEL_join_PMW_grids.rename(columns={"pmw_lat_lon": "lat_lon"}, inplace=True)

# %%
a_year=years[-1]

a_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year, "day_1":"day_365"]
a_dmg = ripser.ripser(a_year_data, maxdim=2)['dgms']
persim.plot_diagrams(a_dmg, show=False, title=f"{a_year}\n rips output\n{diagram_sizes(a_dmg)}", ax=plt.subplot(121))

b_year=years[-2]
b_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==b_year, "day_1":"day_365"]
b_dmg = ripser.ripser(b_year_data, maxdim=2)['dgms']
persim.plot_diagrams(b_dmg, show=False, title=f"{b_year}\n rips output\n{diagram_sizes(a_dmg)}", ax=plt.subplot(122))


# %%
# output dir
output_dir=SNOTEL_dir + "allLocations_aYear_grouped_dgms/"
os.makedirs(output_dir, exist_ok=True)

# %%
for a_year in years:
    a_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year]
    ripser_output = ripser.ripser(a_year_data.loc[:, "day_1":"day_365"], maxdim=2)
    ripser_output["jupyterNotebook_GeneratedThisdata"] = "allLocations_aYear_SNOTEL_PH_Clustering"

    file_Name = str(a_year) + "_" + str(len(a_year_data.station_name.unique())) + "stations_SNOTEL" + ".pkl"
    f = open(output_dir + file_Name, "wb") # create a binary pickle file 
    pickle.dump(ripser_output, f) # write the python object (dict) to pickle file
    f.close() # close file

# %%

# %%
a_year = years[10]
a_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year]
dgms = ripser.ripser(a_year_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms']
persim.plot_diagrams(dgms, show=True, lifetime=True, legend=False)

# %%
# persim.sliced_wasserstein(dgms[1], dgms[1])

# %% [markdown]
# # Form distance matrix

# %%
# %%time
yr_2_yr_H1_distances = pd.DataFrame(columns=years, index=years)

for ii in np.arange(len(years)):
    for jj in np.arange(ii, len(years)):
        ii_year = years[ii]
        jj_year = years[jj]

        ii_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==ii_year]
        jj_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==jj_year]

        ii_dgms_H1 = ripser.ripser(ii_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]
        jj_dgms_H1 = ripser.ripser(jj_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]
        
        yr_2_yr_H1_distances.loc[ii_year, jj_year] = persim.sliced_wasserstein(ii_dgms_H1, jj_dgms_H1)

"""
   Replace NAs with zeros so we can add the dataframe
   to its transpose to get a symmetric matrix
"""
yr_2_yr_H1_distances.fillna(0, inplace=True)

yr_2_yr_H1_distances.loc[:, yr_2_yr_H1_distances.columns]=yr_2_yr_H1_distances.T.values + \
                                                                    yr_2_yr_H1_distances.values

# %%
yr_2_yr_H1_distances_dict={"yr_2_yr_H1_distances":yr_2_yr_H1_distances,
                            "jupyterNotebook_GeneratedThisdata":"allLocations_aYear_SNOTEL_PH_Clustering"
                            }

# %%
file_Name = "year_2_year_H1_distanceMatrix_SNOTEL.pkl"

f = open(output_dir + file_Name, "wb")
pickle.dump(yr_2_yr_H1_distances_dict, f) 
f.close() # close file

# %%
# size = 10
# title_FontSize = 2
# legend_FontSize = 8
# tick_FontSize = 12
# label_FontSize = 14

# params = {'legend.fontsize': 15, # medium, large
#           # 'figure.figsize': (6, 4),
#           'axes.labelsize': size*2,
#           'axes.titlesize': size*1.5,
#           'xtick.labelsize': size*0.00015, #  * 0.75
#           'ytick.labelsize': size, #  * 0.75
#           # 'axes.titlepad':3
#          }

# #
# #  Once set, you cannot change them, unless restart the notebook
# #
# plt.rc('font', family = 'Palatino')
# plt.rcParams['xtick.bottom'] = True
# plt.rcParams['ytick.left'] = True
# plt.rcParams['xtick.labelbottom'] = True
# plt.rcParams['ytick.labelleft'] = True
# plt.rcParams['figure.figsize'] = [15, 4]
# plt.rcParams.update(params)


# %%
params = {'figure.figsize': (10, 4),'axes.titlepad': 10}
plt.rcParams.update(params)

yr_2_yr_H1_distances_array = squareform(yr_2_yr_H1_distances)
yr_2_yr_H1_linkage_matrix = linkage(yr_2_yr_H1_distances_array, "single")
dendrogram(yr_2_yr_H1_linkage_matrix, labels=list(yr_2_yr_H1_distances.columns))
plt.tick_params(axis='both', which='major', labelsize=10)
plt.title("year to year (based on H1 - SNOTEL).")
plt.show()

# %%
yr_2_yr_H1_linkage_matrix

# %%
yr_2_yr_H1_linkage_matrix.shape

# %%
yr_2_yr_H1_linkage_matrix[0:5]

# %%
yr_2_yr_H1_distances.head(5)

# %%
a_year = years[-1]
b_year = years[-2]

aa_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year]
bb_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==b_year]

aa_dgms_H1 = ripser.ripser(aa_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]
bb_dgms_H1 = ripser.ripser(bb_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]

print(f"{persim.sliced_wasserstein(aa_dgms_H1, bb_dgms_H1).round(3)=:}")

persim.plot_diagrams(ripser.ripser(aa_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'], 
                     title=f"{a_year}", ax=plt.subplot(121))

persim.plot_diagrams(ripser.ripser(bb_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'], 
                     title=f"{b_year}", ax=plt.subplot(122))

# %%

# %%
