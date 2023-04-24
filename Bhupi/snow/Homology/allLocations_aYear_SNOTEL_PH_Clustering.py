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
import pickle, h5py

# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag_Others/Bhupi/snow/src/')
import PH as ph
import processing as spr
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
# %%time
all_stations_years_smooth = spr.one_sided_smoothing(all_stations_years, window_size=5)
all_stations_years_smooth.head(2)

# %%
locations = sorted(all_stations_years["station_name"].unique())
years = sorted(all_stations_years["year"].unique())

print (f"{len(locations)=}")

# %%
data_2003 = all_stations_years_smooth[all_stations_years_smooth.year==2003].copy()
len(data_2003.station_name.unique())

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

fig_size_, font_s_=10, 10
params = {"figure.figsize":[fig_size_, fig_size_],"font.size":font_s_}
plt.rcParams.update(params)
persim.plot_diagrams(a_dmg, show=False, title=f"{a_year}, \n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(121))

plt.rcParams.update(params)
persim.plot_diagrams(a_dmg, show=True, title=f"{a_year},\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(122),
                     lifetime=True, legend=False)
plt.tight_layout();
del(a_year, a_year_data, a_dmg)

# %%
# SNOTEL_join_PMW_grids.rename(columns={"pmw_lat_lon": "lat_lon"}, inplace=True)

# %%
a_year, b_year=years[-1], years[-2]

a_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year, "day_1":"day_365"]
a_dmg = ripser.ripser(a_year_data, maxdim=2)['dgms']

b_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==b_year, "day_1":"day_365"]
b_dmg = ripser.ripser(b_year_data, maxdim=2)['dgms']

fig_size_, font_s_=8, 10
params = {"figure.figsize":[fig_size_, fig_size_],"font.size":font_s_}
plt.rcParams.update(params)
persim.plot_diagrams(a_dmg, show=False, title=f"{a_year}\n {ph.diagram_sizes(a_dmg)}", ax=plt.subplot(121))

plt.rcParams.update(params)
persim.plot_diagrams(b_dmg, show=False, title=f"{b_year}\n {ph.diagram_sizes(a_dmg)}", ax=plt.subplot(122))

plt.tight_layout();
del(a_year, a_year_data, a_dmg)
del(b_year, b_year_data, b_dmg)


# %%
# output dir
output_dir=SNOTEL_dir + "allLocs_aYear/"
os.makedirs(output_dir, exist_ok=True)

# %%
for a_year in years:
    a_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year]
    ripser_output = ripser.ripser(a_year_data.loc[:, "day_1":"day_365"], maxdim=2)
    ripser_output["jupyterNotebook_GeneratedThisdata"] = "allLocations_aYear_SNOTEL_PH_Clustering"
    ripser_output["creation_time"] = datetime.now().strftime("%Y_%m_%d_Time_%H_%M")

    file_Name = str(a_year) + "_" + str(len(a_year_data.station_name.unique())) + "stations_SNOTEL" + ".pkl"
    f = open(output_dir + file_Name, "wb") # create a binary pickle file 
    pickle.dump(ripser_output, f) # write the python object (dict) to pickle file
    f.close() # close file
    
del(a_year, a_year_data, ripser_output)

# %%
# persim.sliced_wasserstein(dgms[1], dgms[1])

# %%
params = {'axes.titlepad' : 5,
          'axes.titlesize': 5}
plt.rcParams.update(params)

# persim.sliced_wasserstein(dgms[1], dgms[1])
number_of_cols = int(np.floor(np.sqrt(len(years))))
print (f"{number_of_cols=}")
extra_plots = len(years) - number_of_cols**2
number_of_rows = number_of_cols + int(np.ceil(extra_plots/number_of_cols))
print (f"{number_of_rows=}")

row_count, col_count= 0, 0
subplot_size = 2.5
fig, axs = plt.subplots(number_of_rows, number_of_cols, 
                        figsize=(number_of_cols*subplot_size, number_of_rows*subplot_size),
                        sharey=False, # "col", "row", True, False
                        gridspec_kw={'hspace':0.3, 'wspace':.15})

for a_year in years:
    a_year_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year]
    ripser_output = ripser.ripser(a_year_data.loc[:, "day_1":"day_365"], maxdim=1)
    dgms = ripser_output["dgms"]

    persim.plot_diagrams(dgms, show=False, legend=False, 
                         ax=axs[row_count][col_count])

    axs[row_count][col_count].set(xlabel=None, ylabel=None)
    axs[row_count][col_count].set_title(f"{a_year}", fontdict={"fontsize": 10, "fontweight":"bold"});

    col_count += 1
    if col_count % number_of_cols == 0:
        row_count += 1
        col_count = 0

del(a_year, a_year_data, ripser_output, dgms)
fig_name = output_dir + "ayear_allLocations_SNOTEL_PH" + ".pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')
plt.close('all')

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

yr_2_yr_H1_distances.loc[:, yr_2_yr_H1_distances.columns] = yr_2_yr_H1_distances.T.values + \
                                                                    yr_2_yr_H1_distances.values

del(ii, ii_year, ii_data, ii_dgms_H1)
del(jj, jj_year, jj_data, jj_dgms_H1)

# %%
yr_2_yr_H1_distances_dict={"yr_2_yr_H1_distances":yr_2_yr_H1_distances,
                           "jupyterNotebook_GeneratedThisdata":"allLocations_aYear_SNOTEL_PH_Clustering",
                           "creation_time": datetime.now().strftime("%Y_%m_%d_Time_%H_%M")
                            }
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
import plotly.express as px

length_ = 800
fig = px.imshow(yr_2_yr_H1_distances,
                width=length_, height=length_)

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
)
fig.show()

# %%
# plt.rcParams["figure.figsize"] = [25, 25]
params = {"figure.figsize":[25, 25],
          "axes.titlepad" : 10,
          "axes.titlesize": 30,
          "font.size":15
         }
plt.rcParams.update(params)

plt.pcolor(yr_2_yr_H1_distances)
plt.yticks(np.arange(0.5, len(yr_2_yr_H1_distances.index), 1), yr_2_yr_H1_distances.index)
plt.xticks(np.arange(0.5, len(yr_2_yr_H1_distances.columns), 1), yr_2_yr_H1_distances.columns)
plt.xticks(rotation = 90)
plt.colorbar()
plt.show()

# %%
# plt.rcParams["figure.figsize"] = [25, 25]
params = {"figure.figsize":[10, 10], "font.size" : 10}
a_year, b_year = years[-1], years[-2]

aa_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==a_year]
bb_data = all_stations_years_smooth.loc[all_stations_years_smooth.year==b_year]

aa_dgms_H1 = ripser.ripser(aa_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]
bb_dgms_H1 = ripser.ripser(bb_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'][1]

print(f"{persim.sliced_wasserstein(aa_dgms_H1, bb_dgms_H1).round(3)=:}")

fig_size_, font_s_=8, 10
params = {"figure.figsize":[fig_size_, fig_size_],"font.size":font_s_}
plt.rcParams.update(params)
persim.plot_diagrams(ripser.ripser(aa_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'], 
                     title=f"{a_year}", ax=plt.subplot(121))

plt.rcParams.update(params)
persim.plot_diagrams(ripser.ripser(bb_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'], 
                     title=f"{b_year}", ax=plt.subplot(122))

plt.tight_layout();
del(a_year, aa_data, aa_dgms_H1)
del(b_year, bb_data, bb_dgms_H1)

# %%
