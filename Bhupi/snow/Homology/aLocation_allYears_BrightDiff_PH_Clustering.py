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
import shutup
shutup.please()

# %%
import numpy as np
import pandas as pd

from datetime import date, datetime
import time

import os, os.path, sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

from pylab import imshow
import pickle
import h5py

# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag_Others/Bhupi/snow/src/')
import PH as ph
import processing as sp
import snow_plot_core as spl

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
snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EightyYearsClustering/"
in_dir = snow_TS_dir_base + "Brightness_temperature/"

SNOTEL_dir = snow_TS_dir_base + "SNOTEL_observations/"

# %%
SNOTEL_join_PMW_grids = pd.read_csv(SNOTEL_dir + "SNOTEL_join_PMW_grids.csv")

# %%
file_Name = "all_locs_all_years_eachDayAColumn.pkl"
all_stations_years = pd.read_pickle(in_dir+file_Name)
all_stations_years = all_stations_years["all_locs_all_years_eachDayAColumn"]
all_stations_years.head(2)

# %%
SNOTEL_join_PMW_grids=SNOTEL_join_PMW_grids[["station_name", "pmw_lat_lon"]]
SNOTEL_join_PMW_grids.rename(columns={"pmw_lat_lon": "lat_lon"}, inplace=True)

all_stations_years = pd.merge(all_stations_years, SNOTEL_join_PMW_grids, on=['lat_lon'], how='left')
all_stations_years.drop(columns=['lat_lon'], inplace=True)

# %%
all_stations_years.head(2)

# %% [markdown]
# # Smoothen

# %%
# %%time
all_stations_years_smooth = spr.one_sided_smoothing(all_stations_years, window_size=5)
all_stations_years_smooth.head(2)

# %%
locations = all_stations_years["station_name"].unique()
locations=sorted(locations)
years = all_stations_years["year"].unique()
print (len(locations))

# %%
a_loc = locations[0]
a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
# a_loc_specific_years = a_loc_data.year.unique()
# a_year = a_loc_specific_years[9]
# a_year_data = a_loc_data.loc[a_loc_data.year==a_year]

# ripser.ripser(all_locs_smooth_after_2004[["time_xAxis", "48.97191_-121.05145"]])['dgms']
a_dmg = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"])['dgms']
persim.plot_diagrams(a_dmg, show=False, title=f"{a_loc}\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(121))

persim.plot_diagrams(a_dmg, show=True,  title=f"{a_loc}\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(122),
                     lifetime=True, legend=False)

del(a_loc, a_loc_data, a_dmg)

# %%
a_loc = SNOTEL_join_PMW_grids[SNOTEL_join_PMW_grids.lat_lon=="42.32438_-113.61324"].station_name.values[0]
b_loc = SNOTEL_join_PMW_grids[SNOTEL_join_PMW_grids.lat_lon=="42.69664_-118.61593"].station_name.values[0]

a_loc = "Howell Canyon"
b_loc = "Fish Creek"

a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc, "day_1":"day_365"]
a_dmg = ripser.ripser(a_loc_data, maxdim=1)['dgms']
persim.plot_diagrams(a_dmg, show=False, title=f"{a_loc}, \n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(121))

b_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==b_loc, "day_1":"day_365"]
b_dmg = ripser.ripser(b_loc_data, maxdim=1)['dgms']
persim.plot_diagrams(b_dmg, show=False, title=f"{b_loc},\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(122))

del(a_loc, a_loc_data, a_dmg)
del(b_loc, b_loc_data, b_dmg)

# %%
# output dir
output_dir=in_dir + "aLocation_allYears_grouped_dgms/"
os.makedirs(output_dir, exist_ok=True)

# %%
for a_loc in locations:
    a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
    ripser_output = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"], maxdim=2)
    ripser_output["jupyterNotebook_GeneratedThisdata"] = "aLocation_allYears_BrightDiff_PH_Clustering"
    ripser_output["creation_time"] = datetime.now().strftime("%Y_%m_%d_Time_%H_%M")

    file_Name = a_loc + "_" + str(len(a_loc_data.year.unique())) + "years_BrightDiff" + ".pkl"
    f = open(output_dir + file_Name, "wb") # create a binary pickle file 
    pickle.dump(ripser_output, f) # write the python object (dict) to pickle file
    f.close() # close file
    
del(a_loc, a_loc_data, ripser_output, file_Name)

# %% [markdown]
# # Form a big plot and save to disk

# %%
b_loc = SNOTEL_join_PMW_grids[SNOTEL_join_PMW_grids.lat_lon=="42.69664_-118.61593"].station_name.values[0]
b_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==b_loc, "day_1":"day_365"]
b_dmg = ripser.ripser(b_loc_data, maxdim=2)['dgms']

params = {'axes.titlepad' : 10}
plt.rcParams.update(params)
fig, axs = plt.subplots(1, 2, figsize=(7, 7), sharex=False, sharey=True, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': .2, 'wspace': .3});
(ax1, ax2) = axs;

ax1.grid(False); ax2.grid(False)

persim.plot_diagrams(b_dmg, show=False, ax=ax1)
ax1.set_title(f"{b_loc},\n{ph.diagram_sizes(b_dmg)}", fontdict={"fontsize": 10});

persim.plot_diagrams(b_dmg, show=False, 
                     title=f"{b_loc},\n{ph.diagram_sizes(b_dmg)}", ax=ax2)

ax2.set(xlabel=None);
del(b_loc, b_loc_data, b_dmg)

# %%
params = {'axes.titlepad' : 10,
          'axes.titlesize': 20}
plt.rcParams.update(params)

# persim.sliced_wasserstein(dgms[1], dgms[1])
number_of_cols = int(np.floor(np.sqrt(len(locations))))
print (f"{number_of_cols=}")
extra_plots = len(locations) - number_of_cols**2
number_of_rows = number_of_cols + int(np.ceil(extra_plots/number_of_cols))
print (f"{number_of_rows=}")

row_count, col_count= 0, 0
subplot_size = 3
fig, axs = plt.subplots(number_of_rows, number_of_cols, 
                        figsize=(number_of_cols*subplot_size, number_of_rows*subplot_size),
                        sharey=False, # "col", "row", True, False
                        gridspec_kw={'hspace':0.3, 'wspace':.01})

for a_loc in locations:
    a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
    ripser_output = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"], maxdim=1)
    dgms = ripser_output["dgms"]

    persim.plot_diagrams(dgms, show=False, legend=False, 
                         ax=axs[row_count][col_count])

    axs[row_count][col_count].set(xlabel=None, ylabel=None)
    axs[row_count][col_count].set_title(f"{a_loc}", fontdict={"fontsize": 15});

    col_count += 1
    if col_count % number_of_cols == 0:
        row_count += 1
        col_count = 0

fig_name = output_dir + "aLocation_allYears_BrightDiff_PH" + ".pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')
# plt.close('all')

# %% [markdown]
# # Form distance matrix

# %%
# %%time
loc_2_loc_H1_distances = pd.DataFrame(columns=locations, index=locations)

for ii in np.arange(len(locations)):
    for jj in np.arange(ii, len(locations)):
        ii_loc = locations[ii]
        jj_loc = locations[jj]

        ii_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==ii_loc]
        jj_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==jj_loc]

        ii_dgms_H1 = ripser.ripser(ii_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]
        jj_dgms_H1 = ripser.ripser(jj_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]
        
        loc_2_loc_H1_distances.loc[ii_loc, jj_loc] = persim.sliced_wasserstein(ii_dgms_H1, jj_dgms_H1)

"""Replace NAs with zeros so we can add the dataframe
   to its transpose to get a symmetric matrix
"""
loc_2_loc_H1_distances.fillna(0, inplace=True)

loc_2_loc_H1_distances.loc[:, loc_2_loc_H1_distances.columns]=loc_2_loc_H1_distances.T.values + \
                                                                    loc_2_loc_H1_distances.values

del(ii, ii_loc, ii_data, ii_dgms_H1)
del(jj, jj_loc, jj_data, jj_dgms_H1)

# %%
loc_2_loc_H1_distances_dict={"loc_2_loc_H1_distances":loc_2_loc_H1_distances,
                             "jupyterNotebook_GeneratedThisdata":"aLocation_allYears_BrightDiff_PH_Clustering",
                             "creation_time": datetime.now().strftime("%Y_%m_%d_Time_%H_%M")
                            }

file_Name = "location_2_location_H1_distanceMatrix.pkl"

f = open(output_dir + file_Name, "wb")
pickle.dump(loc_2_loc_H1_distances_dict, f) 
f.close() # close file

# %%
import plotly.express as px

length_ = 800
fig = px.imshow(loc_2_loc_H1_distances,
                width=length_, height=length_)

fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                  paper_bgcolor="LightSteelBlue")
fig.show()

# %%
# plt.rcParams["figure.figsize"] = [25, 25]
params = {"figure.figsize":[25, 25],
          "axes.titlepad" : 10,
          "axes.titlesize": 30,
          "font.size":15
         }
plt.rcParams.update(params)

plt.pcolor(loc_2_loc_H1_distances)
plt.yticks(np.arange(0.5, len(loc_2_loc_H1_distances.index), 1), loc_2_loc_H1_distances.index)
plt.xticks(np.arange(0.5, len(loc_2_loc_H1_distances.columns), 1), loc_2_loc_H1_distances.columns)
plt.xticks(rotation = 90)
plt.colorbar()
plt.show()

# %%
loc_2_loc_H1_distances_dict = pd.read_pickle(output_dir+"location_2_location_H1_distanceMatrix.pkl")
loc_2_loc_H1_distances=loc_2_loc_H1_distances_dict["loc_2_loc_H1_distances"]

# %%
size = 10
title_FontSize = 2
legend_FontSize = 8
tick_FontSize = 12
label_FontSize = 14

params = {'legend.fontsize': 15, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': size*2,
          'axes.titlesize': size*1.5,
          'xtick.labelsize': size*0.00015, #  * 0.75
          'ytick.labelsize': size, #  * 0.75
          'axes.titlepad': 10}
# plt.rc('font', family = 'Palatino')
# plt.rcParams['xtick.bottom'] = True
# plt.rcParams['ytick.left'] = True
# plt.rcParams['xtick.labelbottom'] = True
# plt.rcParams['ytick.labelleft'] = True
plt.rcParams['figure.figsize'] = [15, 4]
plt.rcParams.update(params)

# %%
loc_2_loc_H1_distances_array = squareform(loc_2_loc_H1_distances)
loc_2_loc_H1_linkage_matrix = linkage(loc_2_loc_H1_distances_array, "single")
dendrogram(loc_2_loc_H1_linkage_matrix, labels=list(loc_2_loc_H1_distances.columns))
plt.tick_params(axis='both', which='major', labelsize=10)
plt.title("location to location (based on H1). ")
plt.show()

# %%
plt.rcParams['figure.figsize'] = [7, 3]
a_loc, b_loc = "Howell Canyon", "Fish Creek"

aa_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
bb_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==b_loc]

aa_dgms_H1 = ripser.ripser(aa_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]
bb_dgms_H1 = ripser.ripser(bb_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]
print(f"{persim.sliced_wasserstein(aa_dgms_H1, bb_dgms_H1).round(3)=:}")

persim.plot_diagrams(ripser.ripser(aa_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'], 
                     title=f"{a_loc}", ax=plt.subplot(121))

persim.plot_diagrams(ripser.ripser(bb_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'], 
                     title=f"{b_loc}", ax=plt.subplot(122))

del(a_loc, aa_data, aa_dgms_H1)
del(b_loc, bb_data, bb_dgms_H1)

# %%
