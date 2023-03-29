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
# # Points
#
#    - **Common Database** where all use the same data.
#    - **Identical Processed Data** (so we are on the same page and differences of results to be narrowed down to the algorithms as opposed to working with different datases.).
#    - **Quality control**.
#    - **GitHub**?

# %%
import shutup
shutup.please()

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

sys.path.append('/Users/hn/Documents/00_GitHub/Ag_Others/Bhupi/snow/')
import snow_core as sc

# %% [markdown]
# ### Directories

# %%
snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EithyYearsClustering/"
SNOTEL_dir = snow_TS_dir_base + "SNOTEL_observations/"

temp_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/temparatue/"

# %%
# SNOTEL_join_PMW_grids = pd.read_csv(SNOTEL_dir + "SNOTEL_join_PMW_grids.csv")
# SNOTEL_join_PMW_grids.head(2)

# %%
file_Name = "all_locs_all_years_eachDayAColumn_SNOTEL.pkl"
all_stations_years = pd.read_pickle(SNOTEL_dir+file_Name)
all_stations_years = all_stations_years["all_locs_all_years_eachDayAColumn"]
all_stations_years.head(2)

# %% [markdown]
# # Smoothen

# %% [markdown]
# ### one-sided smoothing

# %%
# %%time
all_locs_years_smooth_1Sided = sc.one_sided_smoothing(all_stations_years, window_size=5)
all_locs_years_smooth_1Sided.head(2)

# %% [markdown]
# ### two-sided smoothing

# %%
# %%time
all_locs_years_smooth_2Sided = sc.two_sided_smoothing(all_stations_years, window_size=5)
all_locs_years_smooth_2Sided.head(2)

# %%
# %%time
all_locs_years_smooth_2Sided_win7=sc.two_sided_smoothing(all_stations_years, window_size=7)
all_locs_years_smooth_2Sided_win7.head(2)

# %% [markdown]
# # Quality Control

# %%
Crater = all_stations_years[all_locs_years_smooth_1Sided.station_name=="Crater Meadows"]
Crater = Crater[Crater.year==2019]

Howell = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Howell Canyon"]
Howell = Howell.loc[Howell.year==2001]

Graham_smooth = all_locs_years_smooth_1Sided[all_locs_years_smooth_1Sided.station_name=="Graham Guard Sta."]
Graham_smooth = Graham_smooth[Graham_smooth.year==2001]

Graham = all_stations_years[all_stations_years.station_name=="Graham Guard Sta."]
Graham = Graham[Graham.year==2001]

subplot_size = 3
fig, axs = plt.subplots(3, 1, figsize=(8, 6),
                        # sharey=True, # "col", "row", True, False
                        gridspec_kw={'hspace':0.5, 'wspace':.15})

axs[0].plot(np.arange(365), Howell.loc[:, "day_1":"day_365"].values[0], 
                    linewidth = 3, ls = '-', label = f'{Howell.year.unique()}');
axs[0].set(xlabel=None, ylabel=None)
axs[0].set_title(f"{Howell.station_name.unique()[0]}", 
              fontdict={"fontsize": 10});
axs[0].legend(loc="best");


axs[1].plot(np.arange(365), Crater.loc[:, "day_1":"day_365"].values[0], 
                    linewidth = 3, ls = '-', label = f'{Crater.year.unique()}');
axs[1].set(xlabel=None, ylabel=None)
axs[1].set_title(f"{Crater.station_name.unique()[0]}", 
              fontdict={"fontsize": 10});
axs[1].legend(loc="best");

axs[2].plot(np.arange(365), 
            Graham_smooth.loc[:, "day_1":"day_365"].values[0], 
            linewidth = 2, ls = "-", label = f"{Graham.year.unique()}, 1-sided smooth", c="dodgerblue");

axs[2].plot(np.arange(365), 
            Graham.loc[:, "day_1":"day_365"].values[0], 
            linewidth = 1.5, ls = "-", label = f"{Graham.year.unique()}", c="r");

axs[2].set(xlabel=None, ylabel=None)
axs[2].set_title(f"Graham", 
              fontdict={"fontsize": 10});
axs[2].legend(loc="upper right");

del(Crater, Howell, Graham_smooth, Graham)

# %%
Graham_temp = pd.read_csv(temp_dir + "Graham_temp.csv")
Graham_temp.head(2)

# %%
len(Graham_temp.max_temp_degF.values)

# %%
Graham_smooth_1Sided = all_locs_years_smooth_1Sided[all_locs_years_smooth_1Sided.station_name=="Graham Guard Sta."]
Graham_smooth_1Sided = Graham_smooth_1Sided[Graham_smooth_1Sided.year==2001]

Graham_smooth_2Sided = all_locs_years_smooth_2Sided[all_locs_years_smooth_2Sided.station_name=="Graham Guard Sta."]
Graham_smooth_2Sided = Graham_smooth_2Sided[Graham_smooth_2Sided.year==2001]

Graham_smooth_2Sided_win7 = all_locs_years_smooth_2Sided_win7[
                                        all_locs_years_smooth_2Sided_win7.station_name=="Graham Guard Sta."]
Graham_smooth_2Sided_win7 = Graham_smooth_2Sided_win7[Graham_smooth_2Sided_win7.year==2001]

Graham = all_stations_years[all_stations_years.station_name=="Graham Guard Sta."]
Graham = Graham[Graham.year==2001]


subplot_size = 3
fig, axs = plt.subplots(1, 1, figsize=(16, 5),
                        # sharey=True, # "col", "row", True, False
                        gridspec_kw={'hspace':0.5, 'wspace':.15})
axs.grid(True);
axs.plot(np.arange(365), 
         Graham.loc[:, "day_1":"day_365"].values[0], 
         linewidth = 3, ls = '-', label = f'{Graham.year.unique()}',  c="g");

axs.plot(np.arange(365), 
         Graham_smooth_1Sided.loc[:, "day_1":"day_365"].values[0], 
         linewidth = 2, ls = "-", label = f"{Graham.year.unique()}, 1-sided smooth", c="dodgerblue");

axs.plot(np.arange(365), 
         Graham_smooth_2Sided.loc[:, "day_1":"day_365"].values[0], 
         linewidth = 1.5, ls = "-", label = f"{Graham.year.unique()}, 2-sided smooth-win5", c="r");

axs.plot(np.arange(365), 
         Graham_smooth_2Sided_win7.loc[:, "day_1":"day_365"].values[0], 
         linewidth = 1.5, ls = "-", label = f"{Graham.year.unique()}, 2-sided smooth-win7", c="k");

#######
####### plot temp
#######
ax2 = axs.twinx() 
ax2.plot(np.arange(365), 
         Graham_temp.max_temp_degF.values, 
         linewidth = 1.5, ls = "-", label = f"Graham temp max");

ax2.plot(np.arange(365), 
         Graham_temp.min_temp_degF.values, 
         linewidth = 1.5, ls = "-", label = f"Graham temp min");

color = 'tab:blue'
ax2.set_ylabel('temp.', color=color)

# axs.fill_between(np.arange(365), Graham_temp.min_temp_degF.values, Graham_temp.max_temp_degF.values)



axs.set(xlabel=None, ylabel=None)
axs.set_title(f"Graham", 
              fontdict={"fontsize": 10});
axs.legend(loc="upper right");

axs.set_ylabel("SNOTEL values", fontsize=12);
axs.set_xlabel("DoY", fontsize=12);

del(Graham_smooth_1Sided, Graham_smooth_2Sided, Graham)

# %%

# %%
locations = all_stations_years["station_name"].unique()
locations=sorted(locations)
years = all_stations_years["year"].unique()
print (f"{len(locations)=}")

# %%
a_loc = locations[0]
a_loc_data = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name==a_loc]

a_dmg = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"])["dgms"]
persim.plot_diagrams(a_dmg, show=False, title=f"{a_loc},\n{sc.diagram_sizes(a_dmg)}", ax=plt.subplot(121))
persim.plot_diagrams(a_dmg, show=True, title=f"{a_loc},\n{sc.diagram_sizes(a_dmg)}", ax=plt.subplot(122),
                     lifetime=True, legend=False)

del(a_loc, a_loc_data, a_dmg)

# %%
a_loc = locations[0]
a_loc_data = all_locs_years_smooth_2Sided.loc[all_locs_years_smooth_2Sided.station_name==a_loc]

a_dmg = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"])["dgms"]
persim.plot_diagrams(a_dmg, show=False, title=f"{a_loc},\n{sc.diagram_sizes(a_dmg)}", ax=plt.subplot(121))
persim.plot_diagrams(a_dmg, show=True, title=f"{a_loc},\n{sc.diagram_sizes(a_dmg)}", ax=plt.subplot(122),
                     lifetime=True, legend=False)

del(a_loc, a_loc_data, a_dmg)

# %%
# SNOTEL_join_PMW_grids.rename(columns={"pmw_lat_lon": "lat_lon"}, inplace=True)

# %%
fig, axs = plt.subplots(1, 2, layout="constrained")

Howell = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Howell Canyon", 
                                          "day_1":"day_365"]

Howell_dmg = ripser.ripser(Howell, maxdim=1)["dgms"]
persim.plot_diagrams(Howell_dmg, show=False, title=f"Howell Canyon,\n {sc.diagram_sizes(Howell_dmg)}", ax=axs[0])

Fish = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Fish Creek", "day_1":"day_365"]
Fish_dmg = ripser.ripser(Fish, maxdim=1)["dgms"]
persim.plot_diagrams(Fish_dmg, show=False, title=f"Fish Creek,\n {sc.diagram_sizes(Fish_dmg)}", ax=axs[1])

axs[1].set(ylabel=None, xlabel=None)
axs[0].set(ylabel=None, xlabel=None)

fig.supxlabel(t="Birth", x=0.5, y=0.1); # common x-axis label
fig.supylabel("Death");                 # common y-axis label

del(Howell, Howell_dmg, Fish, Fish_dmg)


# %%
# remove 2001 from Howell
Howell = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Howell Canyon"]
Howell=Howell.loc[:, "day_1":"day_365"]
Howell_dmg = ripser.ripser(Howell, maxdim=1)["dgms"]

Fish = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Fish Creek", "day_1":"day_365"]
Fish_dmg = ripser.ripser(Fish, maxdim=1)["dgms"]

fig, axs = plt.subplots(1, 2, figsize=(5, 2.5), sharex=False, sharey=False, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.15, 'wspace': .3});
(ax1, ax2) = axs; # ax2
ax1.grid(False); ax2.grid(False);

scatter_size=10
ax1.scatter(Howell_dmg[0][:, 0], Howell_dmg[0][:, 1], s=scatter_size, c="dodgerblue", label=f"$H_0$")
ax1.scatter(Howell_dmg[1][:, 0], Howell_dmg[1][:, 1], s=scatter_size, c="orange", label=f"$H_1$")
ax1.set_title("Howell Canyon")

ax2.scatter(Fish_dmg[0][:, 0], Fish_dmg[0][:, 1], s=scatter_size, c="dodgerblue", label=f"$H_0$")
ax2.scatter(Fish_dmg[1][:, 0], Fish_dmg[1][:, 1], s=scatter_size, c="orange", label=f"$H_1$")
ax2.set_title("Fish Creek")
y_max_fish = max(Fish_dmg[0][:, 1])

ax1.legend(loc="lower right");

ax1.set_xlim([-20, 1100])
ax2.set_xlim([-20, 1100])

ax1.set_ylim([-20, 1100])
ax2.set_ylim([-20, 1100])

ax1.set_ylabel('death');
ax1.set_xlabel('birth');
ax2.set_ylabel('death');
ax2.set_xlabel('birth');

x = np.linspace(0, 1100, 100);
ax1.plot(x, x, linewidth=1.0, linestyle="dashed", c="k");
ax2.plot(x, x, linewidth=1.0, linestyle="dashed", c="k");

ax2.set(ylabel=None);

# del(Howell, Howell_dmg, Fish, Fish_dmg)

# %% [markdown]
# ### Leave 2001 out of Howell Canyon and see the result

# %%
# remove 2001 from Howell
Howell = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Howell Canyon"]
Howell=Howell.loc[:, "day_1":"day_365"]
Howell_dmg = ripser.ripser(Howell, maxdim=1)["dgms"]


Howell = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Howell Canyon"]
Howell_no_2001 = Howell[Howell.year!=2001]
Howell_no_2001 = Howell_no_2001.loc[:, "day_1":"day_365"]
Howell_no_2001_dmg = ripser.ripser(Howell_no_2001, maxdim=1)["dgms"]


Fish = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Fish Creek", "day_1":"day_365"]
Fish_dmg = ripser.ripser(Fish, maxdim=1)["dgms"]

# plot
# fig, axs = plt.subplots(1, 2, layout="constrained", sharey=False)
# persim.plot_diagrams(Howell_dmg, show=False, title=f"Howell Canyon,\n {sc.diagram_sizes(Howell_dmg)}", ax=axs[0])
# persim.plot_diagrams(Fish_dmg, show=False, title=f"Fish Creek,\n {sc.diagram_sizes(Fish_dmg)}", ax=axs[1])
# axs[1].set(ylabel=None, xlabel=None)
# axs[0].set(ylabel=None, xlabel=None)
# fig.supylabel("Death");                 # common y-axis label
# fig.supxlabel(t="Birth", x=0.5, y=0.1); # common x-axis label

fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), sharex=False, sharey=False, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.15, 'wspace': .3});
(ax1, ax2, ax3) = axs; # ax2
ax1.grid(False); ax2.grid(False);

scatter_size=10
ax1.scatter(Howell_dmg[0][:, 0], Howell_dmg[0][:, 1], s=scatter_size, c="dodgerblue", label=f"$H_0$")
ax1.scatter(Howell_dmg[1][:, 0], Howell_dmg[1][:, 1], s=scatter_size, c="orange", label=f"$H_1$")
ax1.set_title("Howell Canyon")

ax2.scatter(Fish_dmg[0][:, 0], Fish_dmg[0][:, 1], s=scatter_size, c="dodgerblue", label=f"$H_0$")
ax2.scatter(Fish_dmg[1][:, 0], Fish_dmg[1][:, 1], s=scatter_size, c="orange", label=f"$H_1$")
ax2.set_title("Fish Creek")


ax3.scatter(Howell_no_2001_dmg[0][:, 0], Howell_no_2001_dmg[0][:, 1], s=scatter_size, c="dodgerblue", label=f"$H_0$")
ax3.scatter(Howell_no_2001_dmg[1][:, 0], Howell_no_2001_dmg[1][:, 1], s=scatter_size, c="orange", label=f"$H_1$")
ax3.set_title("Howell Canyon (no 2001)")

ax1.set_xlim([-20, 1100])
ax2.set_xlim([-20, 1100])
ax3.set_xlim([-20, 1100])

ax1.set_ylim([-20, 1100])
ax2.set_ylim([-20, 1100])
ax3.set_ylim([-20, 1100])

ax1.set_ylabel('death');
ax1.set_xlabel('birth');

ax2.set(ylabel=None);
ax2.set_xlabel('birth');

ax3.set(ylabel=None);
ax3.set_xlabel('birth');

x = np.linspace(0, 1100, 100);
ax1.plot(x, x, linewidth=1.0, linestyle="dashed", c="k");
ax2.plot(x, x, linewidth=1.0, linestyle="dashed", c="k");
ax3.plot(x, x, linewidth=1.0, linestyle="dashed", c="k");

ax1.legend(loc="lower right");
ax2.legend(loc="lower right");
ax3.legend(loc="lower right");

# del(Howell, Howell_dmg, Fish, Fish_dmg)

# %%
print (f"{Howell_no_2001_dmg[0].shape}")
print (f"{Howell_dmg[0].shape}")

print (f"{Howell_no_2001_dmg[1].shape}")
print (f"{Howell_dmg[1].shape}")

# %%
a_loc = "Howell Canyon"
b_loc = "Fish Creek"

Howell_Canyon = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name==a_loc]
Fish_Creek    = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name==b_loc]

curr_data = Fish_Creek.copy()
curr_location = curr_data.station_name.unique()[0]

subplot_size = 3
fig, axs = plt.subplots(2, 1, 
                        figsize=(12, 6),
                        sharey=True, # "col", "row", True, False
                        gridspec_kw={'hspace':0.3, 'wspace':.15})

for a_year in sorted(curr_data.year.unique()):
    a_year_data = curr_data.loc[curr_data.year==a_year]
    
    axs[1].plot(np.arange(365), a_year_data.loc[:, "day_1":"day_365"].values[0], 
                        linewidth = 3, ls = '-', label = f'{a_year}');
    
    axs[1].set(xlabel=None, ylabel=None)
    axs[1].set_title(f"{curr_location}, {curr_data.year.unique().min()} - {curr_data.year.unique().max()}", 
                  fontdict={"fontsize": 10});
    
curr_data = Howell_Canyon.copy()
curr_location = curr_data.station_name.unique()[0]

for a_year in sorted(curr_data.year.unique()):
    a_year_data = curr_data.loc[curr_data.year==a_year]
    
    axs[0].plot(np.arange(365), a_year_data.loc[:, "day_1":"day_365"].values[0], 
                        linewidth = 3, ls = '-', label = f'{a_year}');
    
    axs[0].set(xlabel=None, ylabel=None)
    axs[0].set_title(f"{curr_location}, {curr_data.year.unique().min()} - {curr_data.year.unique().max()}", 
                  fontdict={"fontsize": 10});

# fig_name = output_dir + "FishCreend_HowwellCanyon_SNOTEL.pdf"
# plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')

del(a_loc, b_loc, curr_data, curr_location, a_year, a_year_data)

# %%
params = {"axes.titlepad" : 10,
          "axes.titlesize": 20,
          "axes.titlepad": 10}
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
                        gridspec_kw={"hspace":0.3, "wspace":.01})

for a_loc in locations:
    a_loc_data = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name==a_loc]
    ripser_output = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"], maxdim=1)
    dgms = ripser_output["dgms"]

    persim.plot_diagrams(dgms, show=False, legend=False, 
                         # title=f"{a_loc},\n{sc.diagram_sizes(dgms)}", 
                         ax=axs[row_count][col_count])

    axs[row_count][col_count].set(xlabel=None, ylabel=None)
    axs[row_count][col_count].set_title(f"{a_loc}",  # \n{sc.diagram_sizes(dgms)}
                                          fontdict={"fontsize": 15});

    col_count += 1
    if col_count % number_of_cols == 0:
        row_count += 1
        col_count = 0

del(a_loc, a_loc_data, ripser_output)

# %%
# persim.sliced_wasserstein(dgms[1], dgms[1])

# persim.sliced_wasserstein(dgms[1], dgms[1])
number_of_rows = len(locations)
number_of_cols = 1
print (f"{number_of_rows = }")
print (f"{number_of_cols = }")

row_count, col_count= 0, 0
subplot_size = 3
fig, axs = plt.subplots(number_of_rows, 1, 
                        figsize=(12, number_of_rows*3),
                        sharey=True, # "col", "row", True, False
                        gridspec_kw={"hspace":0.3, "wspace":.01})
loc_count=0
for a_loc in locations:
    a_loc_data = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name==a_loc]

    min_year = a_loc_data.year.unique().min()
    max_year = a_loc_data.year.unique().max()
    curr_location = a_loc_data.station_name.unique()[0]
    for a_year in sorted(a_loc_data.year.unique()):
        a_year_data = a_loc_data.loc[a_loc_data.year==a_year]

        axs[loc_count].plot(np.arange(365), a_year_data.loc[:, "day_1":"day_365"].values[0], 
                            linewidth = 3, ls = "-", label = f"{a_year}");

        axs[loc_count].set(xlabel=None, ylabel=None)
        axs[loc_count].set_title(f"{curr_location}, {min_year} - {max_year}", 
                                  fontdict={"fontsize": 10});
    loc_count+=1

# fig_name = output_dir + "allLocations_allYears_SNOTEL.pdf"
# plt.savefig(fname = fig_name, dpi=100, bbox_inches="tight")

del(a_loc, a_loc_data, min_year, max_year, a_year, a_year_data)

# %%

# %%

# %%
