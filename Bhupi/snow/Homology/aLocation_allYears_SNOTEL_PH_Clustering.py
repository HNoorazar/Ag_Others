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
# In this notebook we collect data of a location across all years in one set (location-2-location comparison). That is a given dataset for which we compute persistent diagram and save it to the disk.

# %%
import shutup
shutup.please()

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
import sys

# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag_Others/Bhupi/snow/src/')
import PH as ph
import processing as spr

# %%
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
# SNOTEL_join_PMW_grids = pd.read_csv(SNOTEL_dir + "SNOTEL_join_PMW_grids.csv")
# SNOTEL_join_PMW_grids.head(2)

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

# %%
locations = sorted(all_stations_years["station_name"].unique())
years = all_stations_years["year"].unique()
print (f"{len(locations)=}")

a_loc = locations[0]
a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
# a_loc_specific_years = a_loc_data.year.unique()
# a_year = a_loc_specific_years[9]
# a_year_data = a_loc_data.loc[a_loc_data.year==a_year]
# ripser.ripser(all_locs_smooth_after_2004[["time_xAxis", "48.97191_-121.05145"]])["dgms"]


a_dmg = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"])["dgms"]

fig_size_, font_s_=8, 10
params = {"figure.figsize":[fig_size_, fig_size_],"font.size":font_s_};
plt.rcParams.update(params);
persim.plot_diagrams(a_dmg, show=False, title=f"{a_loc},\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(121));

plt.rcParams.update(params);
persim.plot_diagrams(a_dmg, show=True, title=f"{a_loc},\n{ph.diagram_sizes(a_dmg)}", ax=plt.subplot(122),
                     lifetime=True, legend=False);

plt.tight_layout();
del(a_loc, a_loc_data, a_dmg)

# %%
# output dir
output_dir=SNOTEL_dir + "aLoc_allYears/"
os.makedirs(output_dir, exist_ok=True)

# %%
a_loc, b_loc = "Howell Canyon", "Fish Creek"

fig_size_, font_s_=6, 10
params = {"figure.figsize":[fig_size_, fig_size_],"font.size":font_s_}
plt.rcParams.update(params)
persim.plot_diagrams(ripser.ripser(all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc, 
                                                                "day_1":"day_365"], maxdim=2)["dgms"], 
                     show=False, title=f"{a_loc}", ax=plt.subplot(121))

plt.rcParams.update(params)
persim.plot_diagrams(ripser.ripser(all_stations_years_smooth.loc[all_stations_years_smooth.station_name==b_loc, 
                                                                  "day_1":"day_365"], maxdim=2)["dgms"], 
                     show=False, title=f"{b_loc}", ax=plt.subplot(122))

plt.tight_layout()
fig_name = output_dir + a_loc.replace(" ", "_") + "_" + b_loc.replace(" ", "_") + "_SNOTEL_PH.pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')

del(a_loc, b_loc, fig_name)


# %%
a_loc, b_loc = "Cougar Mountain", "Cool Creek"
a_loc, b_loc = "Blewett Pass", "Cool Creek"

fig_size_, font_s_=6, 10
params = {"figure.figsize":[fig_size_, fig_size_],"font.size":font_s_}
plt.rcParams.update(params)
persim.plot_diagrams(ripser.ripser(all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc, 
                                                                "day_1":"day_365"], maxdim=2)["dgms"], 
                     show=False, title=f"{a_loc}", ax=plt.subplot(121))

plt.rcParams.update(params)
persim.plot_diagrams(ripser.ripser(all_stations_years_smooth.loc[all_stations_years_smooth.station_name==b_loc, 
                                                                  "day_1":"day_365"], maxdim=2)["dgms"], 
                     show=False, title=f"{b_loc}", ax=plt.subplot(122))

plt.tight_layout()
fig_name = output_dir + a_loc.replace(" ", "_") + "_" + b_loc.replace(" ", "_") + "_SNOTEL_PH.pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')

del(a_loc, b_loc, fig_name)

# %%

# %%
# a_loc, b_loc = "Howell Canyon", "Fish Creek"

# aa_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
# bb_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==b_loc]

# fig_size_, font_s_=6, 10
# params = {"figure.figsize":[fig_size_, fig_size_],"font.size":font_s_}
# plt.rcParams.update(params)
# fig, axs = plt.subplots(1, 2, figsize=(fig_size_, fig_size_),
#                         sharey=False, # "col", "row", True, False
#                         gridspec_kw={'hspace':0.3, 'wspace':.5})

# axs[0] = persim.plot_diagrams(ripser.ripser(aa_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'], 
#                               title=f"{a_loc}", ax=plt.subplot(121))

# plt.rcParams.update(params)
# axs[1] = persim.plot_diagrams(ripser.ripser(bb_data.loc[:, "day_1":"day_365"], maxdim=2)['dgms'], 
#                               title=f"{b_loc}", ax=plt.subplot(122))
# plt.tight_layout()
# del(a_loc, aa_data);
# del(b_loc, bb_data);

# %%

# %%
for a_loc in locations:
    a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
    ripser_output = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"], maxdim=2)
    ripser_output["jupyterNotebook_GeneratedThisdata"] = "aLocation_allYears_SNOTEL_PH_Clustering"
    ripser_output["creation_time"] = datetime.now().strftime("%Y_%m_%d_Time_%H_%M")

    file_Name = a_loc + "_" + str(len(a_loc_data.year.unique())) + "years_SNOTEL" + ".pkl"
    f = open(output_dir + file_Name, "wb") # create a binary pickle file 
    pickle.dump(ripser_output, f) # write the python object (dict) to pickle file
    f.close() # close file
    
del(a_loc, a_loc_data, ripser_output, file_Name)

# %%
params = {"axes.titlepad" : 10,
          "axes.titlesize": 20}
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
    a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
    ripser_output = ripser.ripser(a_loc_data.loc[:, "day_1":"day_365"], maxdim=1)
    dgms = ripser_output["dgms"]

    persim.plot_diagrams(dgms, show=False, legend=False, 
                         ax=axs[row_count][col_count])

    axs[row_count][col_count].set(xlabel=None, ylabel=None)
    axs[row_count][col_count].set_title(f"{a_loc}",
                                          fontdict={"fontsize": 15});
    col_count += 1
    if col_count % number_of_cols == 0:
        row_count += 1
        col_count = 0
plt.tight_layout()
fig_name = output_dir + "aLocation_allYears_SNOTEL_PH" + ".pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches="tight")
# plt.close('all')

del(a_loc, a_loc_data, ripser_output)

# %%
# persim.sliced_wasserstein(dgms[1], dgms[1])
# SNOTEL_join_PMW_grids.rename(columns={"pmw_lat_lon": "lat_lon"}, inplace=True)

# %% [markdown]
# # Form distance matrix

# %%
# %%time
loc_2_loc_H1_distances = pd.DataFrame(columns=locations, index=locations)

for ii in np.arange(len(locations)):
    for jj in np.arange(ii, len(locations)):
        ii_loc, jj_loc = locations[ii], locations[jj]

        ii_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==ii_loc]
        jj_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==jj_loc]

        ii_dgms_H1 = ripser.ripser(ii_data.loc[:, "day_1":"day_365"], maxdim=2)["dgms"][1]
        jj_dgms_H1 = ripser.ripser(jj_data.loc[:, "day_1":"day_365"], maxdim=2)["dgms"][1]
        
        loc_2_loc_H1_distances.loc[ii_loc, jj_loc] = persim.sliced_wasserstein(ii_dgms_H1, jj_dgms_H1)
"""
   Replace NAs with zeros so we can add the dataframe
   to its transpose to get a symmetric matrix
"""
loc_2_loc_H1_distances.fillna(0, inplace=True)
loc_2_loc_H1_distances.loc[:, loc_2_loc_H1_distances.columns]=loc_2_loc_H1_distances.T.values + \
                                                                    loc_2_loc_H1_distances.values

del(ii, ii_loc, ii_data, ii_dgms_H1)
del(jj, jj_loc, jj_data, jj_dgms_H1)

# %%
loc_2_loc_H1_distances_dict={"loc_2_loc_H1_distances":loc_2_loc_H1_distances,
                             "jupyterNotebook_GeneratedThisdata":"aLocation_allYears_SNOTEL_PH_Clustering",
                             "creation_time": datetime.now().strftime("%Y_%m_%d_Time_%H_%M")
                            }

file_Name = "location_2_location_H1_distanceMatrix_SNOTEL.pkl"
f = open(output_dir + file_Name, "wb")
pickle.dump(loc_2_loc_H1_distances_dict, f) 
f.close() # close file

# %%
# size = 10
# title_FontSize = 2
# legend_FontSize = 8
# tick_FontSize = 12
# label_FontSize = 14

# params = {"legend.fontsize": 15, # medium, large
#           # "figure.figsize": (6, 4),
#           "axes.labelsize": size*2,
#           "axes.titlesize": size*1.5,
#           "xtick.labelsize": size*0.00015, #  * 0.75
#           "ytick.labelsize": size, #  * 0.75
#           # "axes.titlepad":3
#          }

# #
# #  Once set, you cannot change them, unless restart the notebook
# #
# # plt.rc("font", family = "Palatino")
# plt.rcParams["xtick.bottom"] = True
# plt.rcParams["ytick.left"] = True
# plt.rcParams["xtick.labelbottom"] = True
# plt.rcParams["ytick.labelleft"] = True
# plt.rcParams["figure.figsize"] = [15, 4]
# plt.rcParams.update(params)

# %%
plt.rcParams["figure.figsize"] = [15, 4]
plt.rcParams.update(params)

loc_2_loc_H1_distances_array = squareform(loc_2_loc_H1_distances)
loc_2_loc_H1_linkage_matrix = linkage(loc_2_loc_H1_distances_array, "single")
dendrogram(loc_2_loc_H1_linkage_matrix, labels=list(loc_2_loc_H1_distances.columns))
plt.tick_params(axis="both", which="major", labelsize=10)
plt.title("location to location (based on H1. Actual Values).", 
          {"fontsize": 15, # plt.rcParams["axes.titlesize"],
           "fontweight" : plt.rcParams["axes.titleweight"],
           "verticalalignment": "baseline",
           "horizontalalignment": "center"}
         )

fig_name = output_dir + "aLocation_allYears_SNOTEL_PH_dend_AV" + ".pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches="tight")

plt.show()

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
          "font.size":15}
plt.rcParams.update(params)

plt.pcolor(loc_2_loc_H1_distances)
plt.yticks(np.arange(0.5, len(loc_2_loc_H1_distances.index), 1), loc_2_loc_H1_distances.index)
plt.xticks(np.arange(0.5, len(loc_2_loc_H1_distances.columns), 1), loc_2_loc_H1_distances.columns)
plt.xticks(rotation = 90)
plt.colorbar()

plt.title("location to location (based on H1. Actual Values).", 
          {"fontsize": 20, # plt.rcParams["axes.titlesize"],
           "fontweight" : plt.rcParams["axes.titleweight"],
           "verticalalignment": "baseline",
           "horizontalalignment": "center"}
         )

fig_name = output_dir + "HeatMap_SNOTEL_aLoc_allYrs.pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# # Re-order

# %%
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

graph = csr_matrix(loc_2_loc_H1_distances)
rcm_ = reverse_cuthill_mckee(graph)
new_order = loc_2_loc_H1_distances.columns[rcm_]

L2L_H1_dist_rcm = loc_2_loc_H1_distances.copy()
L2L_H1_dist_rcm = L2L_H1_dist_rcm[new_order]
L2L_H1_dist_rcm = L2L_H1_dist_rcm.loc[new_order]

# %%
# plt.rcParams["figure.figsize"] = [25, 25]
params = {"figure.figsize":[25, 25],
          "axes.titlepad" : 10,
          "axes.titlesize": 30,
          "font.size":15
         }
plt.rcParams.update(params)

plt.pcolor(L2L_H1_dist_rcm)
plt.yticks(np.arange(0.5, len(L2L_H1_dist_rcm.index), 1), L2L_H1_dist_rcm.index)
plt.xticks(np.arange(0.5, len(L2L_H1_dist_rcm.columns), 1), L2L_H1_dist_rcm.columns)
plt.xticks(rotation = 90)
plt.colorbar()

plt.title("location to location - Reordered by RCM. (based on H1. Actual Values).", 
          {"fontsize": 20, # plt.rcParams["axes.titlesize"],
           "fontweight" : plt.rcParams["axes.titleweight"],
           "verticalalignment": "baseline",
           "horizontalalignment": "center"}
         )
fig_name = output_dir + "HeatMap_SNOTEL_aLoc_allYrs_RCM.pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')

plt.show()

# %% [markdown]
# ### Community Detection (Louvain method NetworkX)
#
#    - [Link](https://python-louvain.readthedocs.io/en/latest/)

# %%
# %load_ext autoreload
# %autoreload

# %%
import community.community_louvain as community_louvain
from communities.algorithms import louvain_method

# %%
# import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

# %%
# graph_L2L_H1_dist = nx.Graph(loc_2_loc_H1_distances)
# graph_L2L_H1_dist.add_weighted_edges_from(loc_2_loc_H1_distances[['source', 'target', 'weight']].values)

# %%
adj_matrix = np.array([[0, 1, 1, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0],
                       [1, 1, 0, 1, 0, 0],
                       [0, 0, 1, 0, 1, 1],
                       [0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 1, 0]])
communities, _ = louvain_method(adj_matrix)
len(communities)

# %%
SNOTEL_communities, _ = louvain_method(loc_2_loc_H1_distances.values)
len(SNOTEL_communities)

# %%

# %%

# %%

# %%
params = {"figure.figsize":[5, 5],
          "axes.titlepad" : 10,
          "axes.titlesize": 30,
          "font.size":15}
plt.rcParams.update(params)

# load the karate club graph
G = nx.karate_club_graph()

#first compute the best partition
partition = community_louvain.best_partition(G)

# draw the graph
pos = nx.spring_layout(G)

# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()

# %%

# %%

# %%

# %%
# curr_data = Howell_Canyon.copy()
# curr_location = curr_data.station_name.unique()[0]

# number_of_rows = len(curr_data.year.unique())
# number_of_columns = 1

# row_count, col_count= 0, 0

# subplot_size = 3
# fig, axs = plt.subplots(number_of_rows, number_of_columns, 
#                         figsize=(12, 3*number_of_rows),
#                         sharey=False, # "col", "row", True, False
#                         gridspec_kw={'hspace':0.3, 'wspace':.15})

# for a_year in sorted(curr_data.year.unique()):
#     a_year_data = curr_data.loc[curr_data.year==a_year]
    
#     axs[row_count].plot(np.arange(365), a_year_data.loc[:, "day_1":"day_365"].values[0], 
#                         linewidth = 3, ls = '-', label = f'{a_year}', c="dodgerblue");
    
#     axs[row_count].set(xlabel=None, ylabel=None)
#     axs[row_count].set_title(f"{curr_location}, {a_year}", fontdict={"fontsize": 10});
#     # axs[row_count].legend(loc="upper right");
#     row_count += 1
        
# # fig_name = output_dir + curr_location + "_SNOTEL" + ".pdf"
# # plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')

# del(curr_data, curr_location, a_year, a_year_data)

# %%

# %%
# curr_data = Howell_Canyon.copy()
# curr_location = curr_data.station_name.unique()[0]

# subplot_size = 3
# fig, axs = plt.subplots(1, 1, 
#                         figsize=(12, 3),
#                         sharey=False, # "col", "row", True, False
#                         gridspec_kw={'hspace':0.3, 'wspace':.15})

# for a_year in sorted(curr_data.year.unique()):
#     a_year_data = curr_data.loc[curr_data.year==a_year]
    
#     axs.plot(np.arange(365), a_year_data.loc[:, "day_1":"day_365"].values[0], 
#                         linewidth = 3, ls = '-', label = f'{a_year}');
    
#     axs.set(xlabel=None, ylabel=None)
#     axs.set_title(f"{curr_location}, {curr_data.year.unique().min()} - {curr_data.year.unique().max()}", 
#                   fontdict={"fontsize": 10});

# # fig_name = output_dir + curr_location + "_SNOTEL" + ".pdf"
# # plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')
# # plt.close('all')

# del(curr_data, curr_location, a_year, a_year_data)

# %%
a_loc, b_loc = "Howell Canyon", "Fish Creek"
# a_loc, b_loc = "Cougar Mountain", "Cool Creek"
# a_loc, b_loc = "Blewett Pass", "Cool Creek"

Howell_Canyon = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]
Fish_Creek    = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==b_loc]

curr_data = Fish_Creek.copy()
curr_location = curr_data.station_name.unique()[0]

fig, axs = plt.subplots(2, 1, figsize=(12, 6),
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

fig_name = output_dir + a_loc.replace(" ", "_") + "_" + b_loc.replace(" ", "_") + "_SNOTEL.pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')

del(a_loc, b_loc, curr_data, curr_location, a_year, a_year_data, fig_name)

# %%
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
    a_loc_data = all_stations_years_smooth.loc[all_stations_years_smooth.station_name==a_loc]

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

fig_name = output_dir + "allLocations_allYears_SNOTEL.pdf"
plt.savefig(fname = fig_name, dpi=100, bbox_inches="tight")

del(a_loc, a_loc_data, min_year, max_year, a_year, a_year_data)
