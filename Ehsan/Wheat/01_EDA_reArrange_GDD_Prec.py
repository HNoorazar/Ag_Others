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

wheat_plot_dir = data_dir_base + "plots/GDD_precip/"
os.makedirs(wheat_plot_dir, exist_ok=True)

# %%

# %%
data_ = pd.read_pickle(wheat_reOrganized + "average_and_seperate_varieties.sav")
data_.keys()

# %%
averaged_varieties = data_["averaged_varieties"]
separate_varieties = data_["separate_varieties"]
dates = data_["dates"]

separate_varieties.head(2)

# %%

# %%
averaged_varieties.head(2)

# %% [markdown]
# # GDD and Precip model
#
# replace NAs in gdd and precip. since after harvest date, they are not measured, but those columns exist because of other location, year combos!

# %%
# grab gdd and precip
x_vars = [x for x in separate_varieties.columns if ("gdd" in x) or ("precip" in x)]

# drop dgdd
x_vars = [x for x in x_vars if not ("dgdd" in x)]
x_vars[:3]

# %%

# %% [markdown]
# ## 4 seasons
#
# 4 season might be reasonable to try?

# %%
wanted_cols = ['location', 'year', 'variety', 'yield'] + x_vars
df = separate_varieties[wanted_cols].copy()
df = pd.merge(df, dates[["year", "location", "season_length"]], on = ["year", "location"], how="left")

# %%
subseason_count = 4
df["season_length_weeks"] = df["season_length"] / 7
df["week_count_per_season"] = round(df["season_length_weeks"] / subseason_count)
df["week_count_per_season"] = df["week_count_per_season"].astype(int)

df.fillna(value=0, inplace=True)
df.head(2)

# %%
dict_season = {"location" : list(df["location"]),
               "year" : list(df["year"]),
               "variety" : list(df["variety"]),
               "yield" : list(df["yield"]),
              }
df_season = pd.DataFrame(dict_season)
df_season.head(2)

# %%
desired_features = ["gdd", "precip"]
# form new season-wise columns
pre_var = ["s" + str(x) + "_" for x in list(range(1, subseason_count+1))]
pre_var = pre_var * len(desired_features)
post_feature = sorted(desired_features * 4)
season_cols = [x + y for x, y in zip(pre_var, post_feature)]

# add new columns to df_season
df_season[season_cols] = -10
df_season.head(2)

# %%
# %%time

# we have to do a for-loop
for idx in df_season.index:
    curr_df = df.loc[idx].copy()
    week_count_per_season = curr_df["week_count_per_season"]
    # print (curr_df["location"] + " - " + str(curr_df["year"]))
    
    # The cuts between each season
    cut_offs = list(week_count_per_season * range(1, subseason_count+1))
    # since there is only 25 weeks worth of data
    # replace the last cutoff to 25 so that last season
    # contains everything.
    cut_offs[-1] = 25
    # takes care of first season
    cut_offs = [0] + cut_offs
    
    for season_ in range(1, subseason_count+1):
        start_week = cut_offs[season_-1] + 1
        end_week   = cut_offs[season_] + 1
        for var_ in desired_features:
            weekly_cols = [str(x) + "_" + var_ for x in range(start_week, end_week)]
            df_season.loc[idx, "s"+str(season_)+"_"+var_] = curr_df[weekly_cols].sum()

df_season.head(2)

# %%
df_season[(df_season["location"] == "Almira") & (df_season["variety"] == "Alpowa")]

# %%
locations = df.location.unique()
varieties = df.variety.unique()

cols_= ["location", "wheat", "start_year", "end_year"]
years_loc_timeSpan = pd.DataFrame(columns = cols_, index = range(len(locations)*len(varieties)))
counter = 0

for a_loc in locations:
    for wheat in varieties:
        A = df[(df["location"] == a_loc) & (df["variety"] == wheat)].copy()
        years_loc_timeSpan.loc[counter, cols_] = [a_loc, wheat, A.year.min(), A.year.max()]
        counter+=1

# %%
A = df[(df["location"] == "Almira") & (df["variety"] == "Alpowa")].copy()
A

# %%
tick_legend_FontSize = 15
params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
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
season_gdd_cols = [x for x in df_season.columns if "gdd" in x]
season_precip_cols = [x for x in df_season.columns if "precip" in x]

# %%
cols_ = ["yield"] + season_gdd_cols + season_precip_cols

my_scatter = sns.pairplot(df_season[cols_], size=2, corner=True, plot_kws={"s": 4})

fig_name = wheat_plot_dir + "4Season" + "_corr.png"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

fig_name = wheat_plot_dir + "4Season" + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

# %%

# %%
cols_ = ["yield"] + season_gdd_cols + season_precip_cols

loc_ = locations[0]
variety = varieties[10]
df_vari = df_season[(df_season["variety"] == variety) & (df_season["location"] == loc_)]


my_scatter = sns.pairplot(df_vari[cols_], size=2, corner=True, plot_kws={"s": 10})

fig_name = wheat_plot_dir + "4Season_" + variety + "_"  + loc_ + "_corr"
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")

# %% [markdown]
# ### for a given variety

# %%
df_season.head(2)

# %%
varieties

# %%
loc_ = locations[0]
variety = varieties[10]
df_vari = df_season[(df_season["variety"] == variety)]
my_scatter = sns.pairplot(df_vari[cols_], size=2, corner=True, plot_kws={"s": 6})

fig_name = wheat_plot_dir + "4Season_" + variety + "_corr"
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")

# %% [markdown]
# ### Not any correlation between yield and seasonal variables. What about annual?
#
# May be correlations occur in higher dimension (i.e. not pairwise vars)?

# %%
dict_season = {"location" : list(df["location"]),
               "year" : list(df["year"]),
               "variety" : list(df["variety"]),
               "yield" : list(df["yield"]),
              }
df_year = pd.DataFrame(dict_season)
df_year.head(2)

# %%
gdd_cols = [x for x in df.columns if "gdd" in x]
precip_cols = [x for x in df.columns if "precip" in x]

df_year["all_gdd"] = df[gdd_cols].sum(axis=1)
df_year["all_precip"] = df[precip_cols].sum(axis=1)
df_year.head(2)

# %%
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
          # 'figure.figsize': (2, 2),
          "axes.labelsize": tick_legend_FontSize * 1,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1,
          "ytick.labelsize": tick_legend_FontSize * 1,
          "axes.titlepad": 10}
plt.rcParams.update(params)

cols_ = ["yield", "all_gdd", "all_precip"]

loc_ = locations[0]
variety = varieties[10]
df_vari = df_year[(df_year["variety"] == variety)]
my_scatter = sns.pairplot(df_vari[cols_], size=1.5, corner=True, plot_kws={"s": 8})

fig_name = wheat_plot_dir + "annual_" + variety + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

# %%
cols_ = ["yield", "all_gdd", "all_precip"]         

loc_ = locations[0]
variety = varieties[2]
df_vari = df_year[(df_year["variety"] == variety) & (df_year["location"] == loc_)]
my_scatter = sns.pairplot(df_vari[cols_], size=1.5, corner=True, plot_kws={"s": 8})

fig_name = wheat_plot_dir + "annual_" + variety + "_" + loc_ + "_corr.pdf"
plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

# %%

# %%
cols_ = ["yield", "all_gdd", "all_precip"]         

loc_ = locations[0]
variety = varieties[2]
df_vari = df_year[(df_year["variety"] == variety)]
my_scatter = sns.pairplot(df_vari[cols_ + ["location"]], hue="location", size=1.5, 
                          corner=True, plot_kws={"s": 8})

fig_name = wheat_plot_dir + "annual_" + variety + "_corr.pdf"
# plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

# %%

# %%
cols_ = ["yield", "all_gdd", "all_precip"]         

loc_ = locations[0]
variety = varieties[2]
df_vari = df_year[(df_year["location"] == loc_)]
my_scatter = sns.pairplot(df_vari[cols_ + ["variety"]], hue="variety", size=1.5, 
                          corner=True, plot_kws={"s": 8})

fig_name = wheat_plot_dir + "annual_" + loc_ + "_corr.pdf"
# plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

# %%

# %% [markdown]
# ### Linear Regression with annual data

# %%
from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda

# %%
df_year.head(2)

# %%
depen_var, indp_vars = "yield", ["all_precip"]

m5 = spreg.OLS_Regimes(y = df_year[depen_var].values, 
                       x = df_year[indp_vars].values, 
                       # Variable specifying neighborhood membership
                       regimes = df_year["variety"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var, # Dependent variable name
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results.transpose()

# %%

# %%
depen_var, indp_vars = "yield", ["all_gdd"]

m5 = spreg.OLS_Regimes(y = df_year[depen_var].values,  x = df_year[indp_vars].values, 
                       regimes = df_year["variety"].tolist(),
                       constant_regi="many",          
                       regime_err_sep=False,
                       name_y=depen_var,
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results.transpose()

# %%

# %%
depen_var, indp_vars = "yield", ["all_gdd", "all_precip"]

m5 = spreg.OLS_Regimes(y = df_year[depen_var].values, x = df_year[indp_vars].values, 
                       regimes = df_year["variety"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(),
                           "Std. Error": m5.std_err.flatten(),
                           "P-Value": [i[1] for i in m5.t_stat],}, 
                          index=m5.name_x)
m5_results.transpose()

# %%

# %%
depen_var, indp_vars = "yield", ["all_gdd", "all_precip"]

m5 = spreg.OLS(y = df_year[depen_var].values, x = df_year[indp_vars].values, 
               name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(),
                           "Std. Error": m5.std_err.flatten(),
                           "P-Value": [i[1] for i in m5.t_stat],}, 
                          index=m5.name_x)
m5_results.transpose()

# %%
df_year.head(2)

# %% [markdown]
# ### Take average of yield per location, year!
# and see if that solves the problem of wide range of yields

# %%
df_year_avg = df_year[["location", "year", "yield"]].copy()
df_year_avg = df_year_avg.groupby(["location", "year"]).mean().reset_index(drop=False)


df_year_weather = df_year[["location", "year", "all_gdd", "all_precip"]].copy()
df_year_weather.drop_duplicates(inplace=True)

df_year_avg = pd.merge(df_year_avg, df_year_weather, on=["location", "year"], how="left")
df_year_avg.head(2)

# %%
cols_ = ["yield", "all_gdd", "all_precip"]         
my_scatter = sns.pairplot(df_year_avg[cols_ ], size=1.5, corner=True, plot_kws={"s": 8})
fig_name = wheat_plot_dir + "annual_" + loc_ + "_corr.pdf"
# plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

# %%
cols_ = ["yield", "all_gdd", "all_precip"]         
loc_ = locations[4]
df_vari = df_year_avg[(df_year_avg["location"] == loc_)]
my_scatter = sns.pairplot(df_vari[cols_ ], size=1.5, corner=True, plot_kws={"s": 8})

fig_name = wheat_plot_dir + "annual_" + loc_ + "_corr.pdf"
# plt.savefig(fname=fig_name, dpi=300, bbox_inches="tight")

# %%

# %% [markdown]
# ## Seasonal Average

# %%
season_gdd_cols = [x for x in df_season if "gdd" in x]
season_precip_cols = [x for x in df_season if "precip" in x]

# %%
df_season_avg = df_season[["location", "year", "yield"]].copy()
df_season_avg = df_season_avg.groupby(["location", "year"]).mean().reset_index(drop=False)
df_season_avg.head(2)

df_season_weather = df_season[["location", "year"] + season_gdd_cols + season_precip_cols].copy()
print (df_season_weather.shape)
df_season_weather.drop_duplicates(inplace=True)
print (df_season_weather.shape)

df_season_avg = pd.merge(df_season_avg, df_season_weather, on=["location", "year"], how="left")
df_season_avg.head(2)

# %%
cols_ = ["yield"] +  season_gdd_cols + season_precip_cols

loc_ = locations[0]
# df_vari = df_season[(df_season["variety"] == variety)]
my_scatter = sns.pairplot(df_season_avg[cols_ + ["location"]], hue="location",
                          size=1.5, corner=True, plot_kws={"s": 6})

sns.move_legend(my_scatter, "upper left", bbox_to_anchor=(0.5, .8))

fig_name = wheat_plot_dir + "4Season_averaged_corr"
plt.savefig(fname=fig_name + ".png", dpi=200, bbox_inches="tight")
plt.savefig(fname=fig_name + ".pdf", dpi=200, bbox_inches="tight")

# %%
