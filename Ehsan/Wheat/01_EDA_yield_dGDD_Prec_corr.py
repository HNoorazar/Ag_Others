# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from datetime import datetime

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag_Others/Ehsan/Wheat/")
import wheat_core as wc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/Ehsan/wheat/"
wheat_reOrganized = data_dir_base + "wheat_reOrganized/"

wheat_plot_dir = data_dir_base + "plots/"
dgdd_plot_dir  = data_dir_base + "plots/dGDD_precip/"
gdd_plot_dir   = data_dir_base + "plots/GDD_precip/"

os.makedirs(wheat_plot_dir, exist_ok=True)
os.makedirs(dgdd_plot_dir, exist_ok=True)
os.makedirs(gdd_plot_dir, exist_ok=True)

# %%
dpi_ = 200

# %%
data_ = pd.read_pickle(wheat_reOrganized + "average_and_seperate_varieties.sav")
list(data_.keys())

# %%
averaged_varieties_weekly = data_["averaged_varieties_weekly"]
separate_varieties_weekly = data_["separate_varieties_weekly"]

separate_varieties_annual = data_["separate_varieties_annual"]
averaged_varieties_annual = data_["averaged_varieties_annual"]

separate_varieties_4season = data_["separate_varieties_4season"]
averaged_varieties_4season = data_["averaged_varieties_4season"]

dates = data_["dates"]

averaged_varieties_weekly.head(2)

# %%
averaged_varieties_weekly.head(2)

# %% [markdown]
# # GDD and Precip model
#
# replace NAs in gdd and precip. since after harvest date, they are not measured, but those columns exist because of other location, year combos!

# %%
separate_varieties_4season[(separate_varieties_4season["location"] == "Almira") & 
                           (separate_varieties_4season["variety"] == "Alpowa")]

# %%
locations = separate_varieties_4season.location.unique()
varieties = separate_varieties_4season.variety.unique()

cols_= ["location", "wheat", "start_year", "end_year"]
years_loc_timeSpan = pd.DataFrame(columns = cols_, index = range(len(locations)*len(varieties)))
counter = 0

for a_loc in locations:
    for wheat in varieties:
        A = separate_varieties_4season[(separate_varieties_4season["location"] == a_loc) & (separate_varieties_4season["variety"] == wheat)].copy()
        years_loc_timeSpan.loc[counter, cols_] = [a_loc, wheat, A.year.min(), A.year.max()]
        counter+=1

# %%
tick_legend_FontSize = 15
params = {"legend.fontsize": tick_legend_FontSize,
          "legend.title_fontsize" : tick_legend_FontSize * 1.3,
          "legend.markerscale" : 2,
          # 'figure.figsize': (6, 4),
          "axes.labelsize": tick_legend_FontSize * 1,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1,
          "ytick.labelsize": tick_legend_FontSize * 1,
          "axes.titlepad": 10}
plt.rcParams.update(params)

# %%
season_gdd_cols = [x for x in separate_varieties_4season.columns if "gdd" in x]
season_gdd_cols = [x for x in season_gdd_cols if not("dgdd" in x)]

season_dgdd_cols = [x for x in separate_varieties_4season.columns if "dgdd" in x]
season_precip_cols = [x for x in separate_varieties_4season.columns if "precip" in x]

# %%
cols_ = ["yield"] + season_dgdd_cols + season_precip_cols

scatt_ = sns.pairplot(separate_varieties_4season[cols_+["variety"]], 
                      hue="variety", diag_kind="hist", 
                      size=2, corner=True, plot_kws={"s": 4})

sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.5, .8))

fig_name = dgdd_plot_dir + "4Season" + "_corr"
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")

# %%
cols_ = ["yield"] + season_gdd_cols + season_precip_cols

scatt_ = sns.pairplot(separate_varieties_4season[cols_+["variety"]], 
                      hue="variety", diag_kind="hist", 
                      size=2, corner=True, plot_kws={"s": 4})

sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.5, .8))

fig_name = gdd_plot_dir + "4Season_corr"
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")

# %%
separate_varieties_4season.head(2)

# %%
cols_ = ["yield"] + season_dgdd_cols + season_precip_cols

loc_ = locations[0]
variety = varieties[10]
df_vari = separate_varieties_4season[(separate_varieties_4season["variety"] == variety) & 
                                     (separate_varieties_4season["location"] == loc_)]

scatt_ = sns.pairplot(df_vari[cols_], size=2, corner=True, plot_kws={"s": 20})

title_ = "location: {}, variety: {}".format(loc_, variety)
scatt_.fig.suptitle(title_, y=.95, fontsize=22)

fig_name = dgdd_plot_dir + "4Season_" + variety + "_"  + loc_ + "_corr"
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")
del(loc_, variety)

# %%

# %%
cols_ = ["yield"] + season_gdd_cols + season_precip_cols

loc_, variety = locations[0], varieties[10]
df_vari = separate_varieties_4season[(separate_varieties_4season["variety"] == variety) & 
                                     (separate_varieties_4season["location"] == loc_)]

scatt_ = sns.pairplot(df_vari[cols_], size=2, corner=True, plot_kws={"s": 20})
title_ = "location: {}, variety: {}".format(loc_, variety)
scatt_.fig.suptitle(title_, y=.95, fontsize=22)

fig_name = gdd_plot_dir + "4Season_" + variety + "_"  + loc_ + "_corr"
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")

del(loc_, variety)

# %%

# %% [markdown]
# ### for a given variety

# %%
variety = varieties[10]
df_vari = separate_varieties_4season[(separate_varieties_4season["variety"] == variety)]
scatt_ = sns.pairplot(df_vari[cols_ + ["location"]], 
                      hue="location", diag_kind="hist", size=2, corner=True, plot_kws={"s": 6})

sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.5, .8))
title_ = "variety: {}".format(variety)
scatt_.fig.suptitle(title_, y=.95, fontsize=25)

file_post = "4Season_" + variety + "_corr"
fig_name = dgdd_plot_dir + file_post
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")
del(variety, df_vari)

# %%

# %%
variety = varieties[10]
df_vari = separate_varieties_4season[(separate_varieties_4season["variety"] == variety)]

scatt_ = sns.pairplot(df_vari[cols_ + ["location"]], hue="location", 
                      diag_kind="hist", size=2, corner=True, plot_kws={"s": 6})

sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.5, .8))
title_ = "variety: {}".format(variety)
scatt_.fig.suptitle(title_, y=.95, fontsize=25)

fig_name = gdd_plot_dir + "4Season_" + variety + "_corr"
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")

del(variety)

# %%

# %% [markdown]
# ### Not any correlation between yield and seasonal variables. What about annual?
#
# May be correlations occur in higher dimension (i.e. not pairwise vars)?

# %%
separate_varieties_annual.head(2)

# %%
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize*0.8,
          "legend.title_fontsize" : tick_legend_FontSize * 1.3,
          "legend.markerscale" : 1.5,
          "axes.labelsize": tick_legend_FontSize * 1,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1,
          "ytick.labelsize": tick_legend_FontSize * 1,
          "axes.titlepad": 10}
plt.rcParams.update(params)

# %%
cols_ = ["yield", "year_dgdd", "year_precip"]+["location"]
variety = varieties[10]
df_vari = separate_varieties_annual[(separate_varieties_annual["variety"] == variety)]
scatt_ = sns.pairplot(df_vari[cols_], hue="location", diag_kind="hist", size=1.8, corner=True, plot_kws={"s": 8})

title_ = "variety: {}".format(variety)
scatt_.fig.suptitle(title_, y=.98) 
sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.58, .93))


fig_name = dgdd_plot_dir + "annual_" + variety + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")
del(variety)

# %%

# %%
cols_ = ["yield", "year_gdd", "year_precip"]+["location"]
variety = varieties[10]
df_vari = separate_varieties_annual[(separate_varieties_annual["variety"] == variety)]
scatt_ = sns.pairplot(df_vari[cols_], hue="location", diag_kind="hist", size=1.8, corner=True, plot_kws={"s": 8})

title_ = "variety: {}".format(variety)
scatt_.fig.suptitle(title_, y=.98) 
sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.58, .93))

fig_name = gdd_plot_dir + "annual_" + variety + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

del(variety)

# %%

# %%
cols_ = ["yield", "year_dgdd", "year_precip"]

loc_, variety = locations[0], varieties[2]
df_vari = separate_varieties_annual[(separate_varieties_annual["variety"] == variety) & 
                                    (separate_varieties_annual["location"] == loc_)]

scatt_ = sns.pairplot(df_vari[cols_], size=1.5, corner=True, plot_kws={"s": 8})

title_ = "location: {}, variety: {}".format(loc_, variety)
scatt_.fig.suptitle(title_, y=1.02, fontsize=12)

fig_name = dgdd_plot_dir + "annual_" + variety + "_" + loc_ + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")
del(loc_, variety)

# %%

# %%
cols_ = ["yield", "year_gdd", "year_precip"]
loc_ = locations[0]
variety = varieties[2]
df_vari = separate_varieties_annual[(separate_varieties_annual["variety"] == variety) & 
                                    (separate_varieties_annual["location"] == loc_)]
scatt_ = sns.pairplot(df_vari[cols_], size=1.5, corner=True, plot_kws={"s": 8})
title_ = "location: {}, variety: {}".format(loc_, variety)
scatt_.fig.suptitle(title_, y=.98) 

fig_name = gdd_plot_dir + "annual_" + variety + "_" + loc_ + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

del(variety, loc_)

# %%

# %%
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize*0.8,
          "legend.title_fontsize" : tick_legend_FontSize * 1.3,
          "legend.markerscale" : 1.5,
          "axes.labelsize": tick_legend_FontSize * 1,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1,
          "ytick.labelsize": tick_legend_FontSize * 1,
          "axes.titlepad": 10}

plt.rcParams.update(params)

# %%
cols_ = ["yield", "year_gdd", "year_precip"] + ["location"]

variety = varieties[2]
df_vari = separate_varieties_annual[(separate_varieties_annual["variety"] == variety)]
scatt_ = sns.pairplot(df_vari[cols_], hue="location", diag_kind="hist", size=1.5, corner=True, plot_kws={"s": 8})

title_ = "variety: {}".format(variety)
scatt_.fig.suptitle(title_, y=.98) 
sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.7, .95))

fig_name = gdd_plot_dir + "annual_" + variety + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")
del(variety)

# %%

# %%
cols_ = ["yield", "year_dgdd", "year_precip"]+["location"]

variety = varieties[2]
df_vari = separate_varieties_annual[(separate_varieties_annual["variety"] == variety)]
scatt_ = sns.pairplot(df_vari[cols_], hue="location", diag_kind="hist", size=1.5, corner=True, plot_kws={"s": 8})

title_ = "variety: {}".format(variety)
scatt_.fig.suptitle(title_, y=.98) 
sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.7, .95))

fig_name = dgdd_plot_dir + "annual_" + variety + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")
del(variety)

# %%

# %% [markdown]
# ### Take average of yield per location, year!
# and see if that solves the problem of wide range of yields

# %%
cols_ = ["yield", "year_gdd", "year_precip"]         
scatt_ = sns.pairplot(averaged_varieties_annual[cols_], size=1.5, corner=True, plot_kws={"s": 8})
fig_name = dgdd_plot_dir + "AvgYield_annual_corr.pdf"
# plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

# %%
cols_ = ["yield", "year_gdd", "year_precip"]         
loc_ = locations[4]
df_vari = averaged_varieties_annual[(averaged_varieties_annual["location"] == loc_)]
scatt_ = sns.pairplot(df_vari[cols_ ], size=1.5, corner=True, plot_kws={"s": 8})

title_ = "location: {}".format(loc_)
scatt_.fig.suptitle(title_, y=.98) 

fig_name = gdd_plot_dir + "AvgYield_annual_" + loc_ + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")
del(loc_)

# %%

# %%
cols_ = ["yield", "year_gdd", "year_precip"] + ["location"]

scatt_ = sns.pairplot(averaged_varieties_annual[cols_], 
                      hue="location", diag_kind="hist",
                      size=1.5, corner=True, plot_kws={"s": 6})

title_ = "yield is averaged over all varieties"
scatt_.fig.suptitle(title_, x=0.5, y=.98, fontsize=12)
sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.7, .9))

fig_name = gdd_plot_dir + "annual_yieldAvg_corr"
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")

# %%

# %% [markdown]
# ## Seasonal Average

# %%
tick_legend_FontSize = 15
params = {"legend.fontsize": tick_legend_FontSize,
          "legend.title_fontsize" : tick_legend_FontSize * 1.3,
          "legend.markerscale" : 2,
          "axes.labelsize": tick_legend_FontSize * 1,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1,
          "ytick.labelsize": tick_legend_FontSize * 1,
          "axes.titlepad": 10}
plt.rcParams.update(params)

# %%
cols_ = ["yield"] +  season_gdd_cols + season_precip_cols + ["location"]

loc_ = locations[0]
# df_vari = averaged_varieties_4season[(averaged_varieties_4season["location"] == loc_)]
scatt_ = sns.pairplot(averaged_varieties_4season[cols_], 
                      hue="location", diag_kind="hist",
                      size=1.5, corner=True, plot_kws={"s": 6})

title_ = "yield is averaged over all varieties"
scatt_.fig.suptitle(title_, x=0.4, y=.95, fontsize=22)
sns.move_legend(scatt_, "upper left", bbox_to_anchor=(0.5, .9))

fig_name = gdd_plot_dir + "4Season_averaged_corr"
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")

# %%
