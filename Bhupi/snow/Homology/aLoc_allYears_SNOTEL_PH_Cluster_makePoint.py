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
#    - **Quality control**.
#    - **Identical Processed Data** (so we are on the same page and differences of results to be narrowed down to the algorithms as opposed to working with different datases.).
#    - **Calendar year?**.
#    - **GitHub**?

# %%
import shutup
shutup.please()
# %load_ext autoreload

# %%
import numpy as np
import pandas as pd

from datetime import date, datetime
import time

import os, os.path

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
import ripser
from ripser import Rips #, ripser

import persim
# from persim import plot_diagrams

import tadasets
import kmapper as km # Import the class

# %%
import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag_Others/Bhupi/snow/src/')
import PH as ph
import processing as spr
import snow_plot_core as spl

# %% [markdown]
# ### Directories

# %%
snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EightyYearsClustering/"
SNOTEL_dir = snow_TS_dir_base + "SNOTEL_observations/"

temperature_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/temparatue/"

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
# ### one- and two-sided smoothing

# %%
# %%time
all_locs_years_smooth_1Sided = spr.one_sided_smoothing(all_stations_years, window_size=5)

## two sided smoothing
all_locs_years_smooth_2Sided = spr.two_sided_smoothing(all_stations_years, window_size=5)
all_locs_years_smooth_2Sided_win7=spr.two_sided_smoothing(all_stations_years, window_size=7)

# %% [markdown]
# # Quality Control

# %%
Crater = all_locs_years_smooth_1Sided[all_locs_years_smooth_1Sided.station_name=="Crater Meadows"]
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

axs[2].plot(np.arange(365), Graham_smooth.loc[:, "day_1":"day_365"].values[0], 
            linewidth = 2, ls = "-", label = f"{Graham.year.unique()}, 1-sided smooth", c="dodgerblue");

axs[2].plot(np.arange(365), Graham.loc[:, "day_1":"day_365"].values[0], 
            linewidth = 1.5, ls = "-", label = f"{Graham.year.unique()}", c="r");

axs[2].set(xlabel=None, ylabel=None)
axs[2].set_title(f"Graham", 
              fontdict={"fontsize": 10});
axs[2].legend(loc="upper right");

del(Crater, Howell, Graham_smooth, Graham)

# %%
# # It is linear-interpolation.
# raw_SNOTEL=pd.read_csv(SNOTEL_dir+"89_SNOTEL_stations.csv")
# crater_raw = raw_SNOTEL.loc[:, ["Date", "Year", "Crater Meadows"]]
# crater_raw_2019 = crater_raw[crater_raw.Year==2019]
# crater_raw_2019.reset_index(inplace=True, drop=True)
# print (crater_raw_2019.loc[50,"Crater Meadows"]-crater_raw_2019.loc[51,"Crater Meadows"])
# print (crater_raw_2019.loc[51,"Crater Meadows"]-crater_raw_2019.loc[52,"Crater Meadows"])
# print (crater_raw_2019.loc[52,"Crater Meadows"]-crater_raw_2019.loc[53,"Crater Meadows"])
# crater_raw_2019.loc[50:70,]

# %%

# %%
Graham_temp = pd.read_csv(temperature_dir + "Graham_temp.csv")
Graham_temp.head(2)

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
axs.plot(np.arange(365), Graham.loc[:, "day_1":"day_365"].values[0], 
         linewidth = 3, ls = '-', label = f'{Graham.year.unique()}',  c="g");

axs.plot(np.arange(365), Graham_smooth_1Sided.loc[:, "day_1":"day_365"].values[0], 
         linewidth = 2, ls = "-", label = f"{Graham.year.unique()}, 1-sided smooth", c="dodgerblue");

axs.plot(np.arange(365), Graham_smooth_2Sided.loc[:, "day_1":"day_365"].values[0], 
         linewidth = 1.5, ls = "-", label = f"{Graham.year.unique()}, 2-sided smooth-win5", c="r");

axs.plot(np.arange(365), 
         Graham_smooth_2Sided_win7.loc[:, "day_1":"day_365"].values[0], 
         linewidth = 1.5, ls = "-", label = f"{Graham.year.unique()}, 2-sided smooth-win7", c="k");
#######
####### plot temp
#######
ax2 = axs.twinx() 
ax2.plot(np.arange(365), Graham_temp.max_temp_degF.values, 
         linewidth = 1.5, ls = "-", label = f"Graham temp max");

ax2.plot(np.arange(365), Graham_temp.min_temp_degF.values, 
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
locations = all_stations_years["station_name"].unique()
locations=sorted(locations)
years = all_stations_years["year"].unique()
print(f"{len(locations)=}")

# %% [markdown]
# ### 1 sided and 2 sided smoothing. Not much of a difference

# %%
params = {"figure.figsize":[8, 8],
          "font.size" : 8}
plt.rcParams.update(params)

a_loc = locations[0]
a_loc_data = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name==a_loc]

a_dmg = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"])["dgms"]
persim.plot_diagrams(a_dmg, show=False, title=f"{a_loc},\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(121))
# persim.plot_diagrams(a_dmg, show=True, title=f"{a_loc},\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(122),
#                      lifetime=True, legend=False)

del(a_loc, a_loc_data, a_dmg)

a_loc = locations[0]
a_loc_data = all_locs_years_smooth_2Sided.loc[all_locs_years_smooth_2Sided.station_name==a_loc]
a_dmg = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"])["dgms"]
persim.plot_diagrams(a_dmg, show=False, title=f"{a_loc},\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(122))
del(a_loc, a_loc_data, a_dmg)

# %% [raw]
# fig, axs = plt.subplots(1, 2, layout="constrained")
# Howell = all_locs_years_smooth_1Sided.loc[
#     all_locs_years_smooth_1Sided.station_name=="Howell Canyon", "day_1":"day_365"]
#
# Howell_dmg = ripser.ripser(Howell, maxdim=1)["dgms"]
# persim.plot_diagrams(Howell_dmg, show=False, title=f"Howell Canyon,\n {ph.diagram_sizes(Howell_dmg)}", ax=axs[0])
#
# Fish = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Fish Creek", "day_1":"day_365"]
# Fish_dmg = ripser.ripser(Fish, maxdim=1)["dgms"]
# persim.plot_diagrams(Fish_dmg, show=False, title=f"Fish Creek,\n {ph.diagram_sizes(Fish_dmg)}", ax=axs[1])
#
# axs[1].set(ylabel=None, xlabel=None)
# axs[0].set(ylabel=None, xlabel=None)
#
# fig.supxlabel(t="Birth", x=0.5, y=0.1); # common x-axis label
# fig.supylabel("Death");                 # common y-axis label
#
# del(Howell, Howell_dmg, Fish, Fish_dmg)


# %%
Howell = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Howell Canyon"]
Howell=Howell.loc[:, "day_1":"day_365"]
Howell_dmg = ripser.ripser(Howell, maxdim=1)["dgms"]

Fish = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Fish Creek", "day_1":"day_365"]
Fish_dmg = ripser.ripser(Fish, maxdim=1)["dgms"]

fig, axs = plt.subplots(1, 2, figsize=(6, 2.5), sharex=False, sharey=False, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.15, 'wspace': .3});
ax_min_=-50
ax_max_=round(Howell_dmg[0][:, 1][-2]+Fish_dmg[0][:, 1][-2]*0.1)
spl.plot_aDMG_maxDim2(dgm=Howell_dmg, ax=axs[0], ax_min=ax_min_, ax_max=ax_max_, title_="Howell Canyon")

ax_max_=round(Fish_dmg[0][:, 1][-2]+Fish_dmg[0][:, 1][-2]*0.1)
spl.plot_aDMG_maxDim2(dgm=Fish_dmg, ax=axs[1],   ax_min=ax_min_, ax_max=ax_max_, title_="Fish Creek")

fig.supxlabel(t="Birth", x=0.5, y=-0.1); # common x-axis label
fig.supylabel("Death");                 # common y-axis label

# %% [markdown]
# ### Leave 2001 out of Howell Canyon and see the result

# %% [markdown]
# #### My plot

# %%
# %autoreload
import PH as ph
import processing as sp
import snow_plot_core as spl

Howell = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Howell Canyon"]
Howell=Howell.loc[:, "day_1":"day_365"]
Howell_dmg = ripser.ripser(Howell, maxdim=1)["dgms"]

Howell = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Howell Canyon"]
Howell_no_2001 = Howell[Howell.year!=2001]
Howell_no_2001 = Howell_no_2001.loc[:, "day_1":"day_365"]
Howell_no_2001_dmg = ripser.ripser(Howell_no_2001, maxdim=1)["dgms"]

Fish = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name=="Fish Creek", "day_1":"day_365"]
Fish_dmg = ripser.ripser(Fish, maxdim=1)["dgms"]

#################################################################
fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), sharex=False, sharey=False, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.15, 'wspace': .3});
ax_min_=-20
ax_max_=1400

ax_max_=round(Howell_dmg[0][:, 1][-2]+Fish_dmg[0][:, 1][-2]*0.1)
spl.plot_aDMG_maxDim2(dgm=Howell_dmg, ax=axs[0], ax_min=ax_min_, ax_max=ax_max_, title_="Howell Canyon")

ax_max_=round(Howell_no_2001_dmg[0][:, 1][-2]+Fish_dmg[0][:, 1][-2]*0.1)
spl.plot_aDMG_maxDim2(dgm=Howell_no_2001_dmg, ax=axs[1], ax_min=ax_min_, ax_max=ax_max_, 
                      title_="Howell Canyon (no 2001)")

ax_max_=round(Fish_dmg[0][:, 1][-2]+Fish_dmg[0][:, 1][-2]*0.1)
spl.plot_aDMG_maxDim2(dgm=Fish_dmg, ax=axs[2], ax_min=ax_min_, ax_max=ax_max_, title_="Fish Creek")

# %% [markdown]
# #### persim plot

# %% [raw]
# fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), sharex=False, sharey=False, # sharex=True, sharey=True,
#                        gridspec_kw={'hspace': 0.15, 'wspace': .3}); # layout="constrained"
#
# persim.plot_diagrams(Howell_dmg, show=False, 
#                      title=f"Howell Canyon,\n {ph.diagram_sizes(Howell_dmg)}", ax=axs[0])
# persim.plot_diagrams(Howell_no_2001_dmg, show=False, 
#                      title=f"Howell Canyon (no 2001),\n {ph.diagram_sizes(Howell_no_2001_dmg)}", ax=axs[1])
# persim.plot_diagrams(Fish_dmg, show=False, 
#                      title=f"Fish Creek,\n {ph.diagram_sizes(Fish_dmg)}", ax=axs[2])
#
# axs[2].set(ylabel=None, xlabel=None)
# axs[1].set(ylabel=None, xlabel=None)
# axs[0].set(ylabel=None, xlabel=None)
#
# fig.supxlabel(t="Birth", x=0.5, y=-.2); # common x-axis label
# fig.supylabel("Death");                 # common y-axis label
#
# del(Howell, Howell_dmg, Fish, Fish_dmg, Howell_no_2001, Howell_no_2001_dmg)

# %%
a_loc, b_loc  = "Howell Canyon", "Fish Creek"
Howell_Canyon = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name==a_loc]
Fish_Creek    = all_locs_years_smooth_1Sided.loc[all_locs_years_smooth_1Sided.station_name==b_loc]

curr_data = Fish_Creek.copy()
curr_location = curr_data.station_name.unique()[0]

subplot_size = 3
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharey=True, # "col", "row", True, False
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
# fig_name = output_dir + "FishCreend_HowellCanyon_SNOTEL.pdf"
# plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')
del(a_loc, b_loc, curr_data, curr_location, a_year, a_year_data)

# %%

# %%

# %%
