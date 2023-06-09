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
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

from pylab import imshow
import pickle
import h5py
import sys

# %%
# from jupytext.config import global_jupytext_configuration_directories
# list(global_jupytext_configuration_directories())

# from jupytext.config import find_jupytext_configuration_file
# find_jupytext_configuration_file('/Users/hn/.config/jupytext')

# %%
# !pip3 install --upgrade numpy

# %%

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


# %%
def diagram_sizes(dgms):
    return ", ".join([f"|$H_{i}$|={len(d)}" for i, d in enumerate(dgms)])


# %% [markdown]
# ### Clustering Approaches
#
# It seems we can use usual clustering methods with features generated by persistent homology. TDA itself has clustering algorithms (e.g. Mapper and ToMATo) as well.

# %%
snow_TS_dir_base = "/Users/hn/Documents/data/EithyYearsClustering/"
# snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EithyYearsClustering/"
diff_dir = snow_TS_dir_base + "Brightness_temperature/Only_for_SNOTEL_grids/"

# %%
all_locs_all_years_but_2003 = pd.read_csv(snow_TS_dir_base + \
                                          "Brightness_temperature/" + \
                                          "all_locs_all_years_but_2003.csv")
all_locs_all_years_but_2003.date = pd.to_datetime(all_locs_all_years_but_2003.date)
all_locs_all_years_but_2003.head(2)

# %%
all_locs_after_2004 = all_locs_all_years_but_2003[all_locs_all_years_but_2003.year>=2004].copy()
all_locs_after_2004.reset_index(drop=True, inplace=True)

# %%
locations = list(all_locs_after_2004.columns)

bad_cols = ["month", "day", "year", "date"]
for a_bad in bad_cols:
    locations.remove(a_bad)

print (f"{len(locations)=}")

# %%
# Check if our data is daily
print (len(all_locs_after_2004))
print ((all_locs_after_2004.year.max()-all_locs_after_2004.year.min()+1)*365)

# %% [markdown]
# ### Check window size for moving averages

# %%
a_signal = all_locs_after_2004.loc[all_locs_after_2004.year==2004, all_locs_after_2004.columns[0]]

window_10 = 10
weights_10 = np.arange(1, window_10+1)

window_7 = 7
weights_7 = np.arange(1, window_7+1)

window_3 = 3
weights_3 = np.arange(1, window_3+1)

window_5 =5
weights_5 = np.arange(1, window_5+1)

wma_10 = a_signal.rolling(window_10).apply(lambda a_signal: np.dot(a_signal, weights_10)/weights_10.sum(), raw=True)
wma_7 = a_signal.rolling(window_7).apply(lambda a_signal: np.dot(a_signal, weights_7)/weights_7.sum(), raw=True)
wma_5 = a_signal.rolling(window_5).apply(lambda a_signal: np.dot(a_signal, weights_5)/weights_5.sum(), raw=True)
wma_3 = a_signal.rolling(window_3).apply(lambda a_signal: np.dot(a_signal, weights_3)/weights_3.sum(), raw=True)

fig, ax2 = plt.subplots(1, 1, figsize=(15, 5), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(a_signal, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
# ax2.plot(wma_10, linewidth = 4, ls = '-', label = 'wma_10', c="k");
ax2.plot(wma_7, linewidth = 4, ls = '-', label = 'WMA_7', c="r");
ax2.plot(wma_5, linewidth = 2, ls = '-', label = 'WMA_5', c="k");
ax2.plot(wma_3, linewidth = 2, ls = '-', label = 'WMA_3', c="g");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

# %% [markdown]
# ### Smooth the dataframe by window size 5.

# %%
all_locs_smooth_after_2004 = all_locs_after_2004.copy()

window_5 = 10
weights_5 = np.arange(1, window_5+1)

for a_loc in locations:
    a_signal = all_locs_smooth_after_2004[a_loc]
    wma_5 = a_signal.rolling(window_5).apply(lambda a_signal: np.dot(a_signal, weights_5)/weights_5.sum(), raw=True)
    all_locs_smooth_after_2004[a_loc] = wma_5

all_locs_smooth_after_2004.head(10)

# %% [markdown]
# We lost some data at the beginning due to rolling window. So, we replace them here:

# %%
end = window_5-1
all_locs_smooth_after_2004.iloc[0:end, 0:len(locations)]=all_locs_smooth_after_2004.iloc[end, 0:len(locations)]

# all_locs_smooth_after_2004 = all_locs_smooth_after_2004.assign(time_xAxis=range(len(all_locs_smooth_after_2004)))
all_locs_smooth_after_2004.head(2)

# %%
# In the following I use [ripser.ripser(data_clean)['dgms']]
# as opposed to [rips.transform(data_clean)] since "dgms" is reminder of what we are doing!
# transform is too general!

# %%
years = all_locs_smooth_after_2004.year.unique()

a_year = years[9]
a_loc = locations[0]
a_year_data = all_locs_smooth_after_2004.loc[all_locs_smooth_after_2004.year==a_year, a_loc]
a_year_data_2D = np.concatenate((np.array(a_year_data).reshape(-1,1), 
                                 np.array(range(len(a_year_data))).reshape(-1,1)), 
                                 axis=1)
# ripser.ripser(all_locs_smooth_after_2004[["time_xAxis", "48.97191_-121.05145"]])['dgms']
a_dmg = ripser.ripser(a_year_data_2D)['dgms']

# %%
# plt.plot(dgm_noisy_0_x, dgm_noisy_0_y, 'g^', dgm_noisy_1_x, dgm_noisy_1_y, 'bs');
fig, axs = plt.subplots(1, 1, figsize=(10, 2.5), sharex=False, sharey=False, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.15, 'wspace': .15});
(ax1) = axs; # ax2
ax1.grid(True); # ax2.grid(True);

ax1.plot(list(a_year_data), linewidth = 2, ls = '-', c="dodgerblue", label="brightness diff.")
ax1.set_title(f"brightness diff. (Smoothed, wma ={window_5})")
ax1.legend(loc="lower right");
ax1.set_ylabel('brightness diff.');
ax1.set_xlabel('day of year');

# %%
# plt.plot(dgm_noisy_0_x, dgm_noisy_0_y, 'g^', dgm_noisy_1_x, dgm_noisy_1_y, 'bs');
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=False, sharey=False, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.15, 'wspace': .15});
(ax1) = axs; # ax2
ax1.grid(True); ax2.grid(True);

scatter_size=5
ax1.scatter(a_dmg[0][:, 0], a_dmg[0][:, 1], s=scatter_size, c="r", label="first array of ripser")
ax1.set_title("ripser output")
ax1.legend(loc="lower right");
ax1.set_xlim([-0.1, np.ceil(a_dmg[0][:, 1][-2])+0.1])
ax1.set_ylim([-0.1, np.ceil(a_dmg[0][:, 1][-2])+0.1])

ax1.set_ylabel('death');
ax1.set_xlabel('birth');

# %%
persim.plot_diagrams(a_dmg, show=False, title=f"rips output\n{diagram_sizes(a_dmg)}")

# %%
np.random.seed(10)
data = np.random.random((100,2))
diagrams = ripser.ripser(data)['dgms']
persim.plot_diagrams(diagrams, show=True)

# The following does the same thing as above
# rips = Rips()
# rips.plot(diagrams)

# %% [markdown]
# # Tutorial
# for the rest of this notebook the tutorial is from [https://ripser.scikit-tda.org/en/latest/notebooks/Basic%20Usage.html](https://ripser.scikit-tda.org/en/latest/notebooks/Basic%20Usage.html)

# %%
from sklearn import datasets
data = datasets.make_circles(n_samples=100)[0] + 5 * datasets.make_circles(n_samples=100)[0]

# %%
dgms = ripser.ripser(data)['dgms']
persim.plot_diagrams(dgms, show=True)

# %%
persim.plot_diagrams(dgms, plot_only=[0], ax=plt.subplot(121))
persim.plot_diagrams(dgms, plot_only=[1], ax=plt.subplot(122))

# %%
# I do not see any difference here~
coeff_3 = 3
coeff_7 = 7
coeff_17= 17

dgms_3 = ripser.ripser(data, coeff=coeff_3)['dgms']
dgms_7 = ripser.ripser(data, coeff=coeff_7)['dgms']
dgms_17 = ripser.ripser(data, coeff=coeff_17)['dgms']

persim.plot_diagrams(dgms_3,  plot_only=[1], title=f"Homology of Z/{coeff_3}Z",  ax=plt.subplot(131))
persim.plot_diagrams(dgms_7,  plot_only=[1], title=f"Homology of Z/{coeff_7}Z",  ax=plt.subplot(132))
persim.plot_diagrams(dgms_17, plot_only=[1], title=f"Homology of Z/{coeff_17}Z", ax=plt.subplot(133))

# %%
(dgms_3[1]==dgms_7[1]).sum()==(dgms_3[1].shape[0]*dgms_3[1].shape[1])

# %%
dgms_3[1]

# %%
ripser.ripser(data, coeff=coeff_3).keys()

# %% [markdown]
# #### Specify which homology classes to compute
# We can compute any order of homology, $H_0$, $H_1$, $\dots$. By default, we only compute $H_0$ and $H_1$. 
# You can specify a larger by supplying the argument ```maxdim=p```. It practice, anything above $H_1$ is very slow.

# %%
dgms = ripser.ripser(data, maxdim=2)['dgms']
persim.plot_diagrams(dgms, show=True)

# %% [markdown]
# #### Specify maximum radius for Rips filtration
#
# We can restrict the maximum radius of the VR complex by supplying the argument ```thresh=r```. Certain classes will not be born if their birth time is under the threshold, and other classes will have infinite death times if their ordinary death time is above the threshold

# %%
dgms_thresh_point2 = ripser.ripser(data, thresh=0.2)['dgms']
dgms_thresh_1 = ripser.ripser(data, thresh=1)['dgms']
dgms_thresh_2 = ripser.ripser(data, thresh=2)['dgms']
dgms_thresh_999 = ripser.ripser(data, thresh=999)['dgms']


persim.plot_diagrams(dgms_thresh_point2,  title=f"thresh 0.2", ax=plt.subplot(221))
persim.plot_diagrams(dgms_thresh_1,       title=f"thresh 1",   ax=plt.subplot(222))

persim.plot_diagrams(dgms_thresh_2,       title=f"thresh 2",   ax=plt.subplot(223))
persim.plot_diagrams(dgms_thresh_999,     title=f"thresh 999", ax=plt.subplot(224))

# %%
persim.plot_diagrams(dgms_thresh_999, lifetime=True, ax=plt.subplot(121))
persim.plot_diagrams(dgms_thresh_999, lifetime=False, ax=plt.subplot(122))

# %%
import kmapper as km
from kmapper import jupyter # Creates custom CSS full-size Jupyter screen

# %%
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=0)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0,1]); # X-Y axis

# Create a cover with 10 elements
cover = km.Cover(n_cubes=10)

# %%
# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, cover=cover)

# Visualize it
# _ = mapper.visualize(graph, path_html="output/make_circles_keplermapper_output.html",
#                     title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")

## Uncomment the below to view the above-generated visualization
# jupyter.display(path_html="output/make_circles_keplermapper_output.html")

## Alternatively, use an IFrame to display a vis with a set width and height
#
# from IPython.display import IFrame
# IFrame(src="http://mlwave.github.io/tda/word2vec-gender-bias.html", width=800, height=600)


# %%
import kmapper as km

# Some sample data
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3);

# Initialize
mapper = km.KeplerMapper(verbose=0)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0,1]); # X-Y axis

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, cover=km.Cover(n_cubes=10));

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)");

# %%
VI_idx = "V"
smooth_type = "N"
SR = 3
print (f"Passed Args. are: {VI_idx=:}, {smooth_type=:}, and {SR=:}!")

# %%
variable=""
print(f"{variable:=^100}")

# %%
