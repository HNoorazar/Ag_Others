# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
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
import pymannkendall as mk
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
# plt.rc("font", family="Times")

from datetime import datetime

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag_Others/Ehsan/Wheat/")
import wheat_core as wc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/Ehsan/wheat/"
wheat_reOrganized = data_dir_base + "wheat_reOrganized/"

wheat_plot_dir = data_dir_base + "plots/"
os.makedirs(wheat_plot_dir, exist_ok=True)

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

separate_varieties_annual.head(2)

# %% [markdown]
# ### GDD Spearman test

# %%
yield_TS = separate_varieties_annual["yield"].values
GDD_TS = separate_varieties_annual["year_gdd"].values
Spearman, p_valueSpearman = stats.spearmanr(GDD_TS, yield_TS)
print (Spearman.round(3), p_valueSpearman.round(7))

# %%
DF = separate_varieties_annual.copy() # make the lines shorter!
GDD_spear = DF["variety"].copy()
GDD_spear.drop_duplicates(inplace=True)

spearman_test_cols = ["spearman", "p_valSpearman"]
GDD_spear = pd.concat([GDD_spear, pd.DataFrame(columns = spearman_test_cols)])
GDD_spear[spearman_test_cols] = [-666, -666] * (len(spearman_test_cols)-1)

for a_variety in GDD_spear["variety"].unique():
    yield_TS = DF.loc[DF["variety"]==a_variety, "yield"].values
    GDD_TS = DF.loc[DF["variety"]==a_variety, "year_gdd"].values
    Spearman, p_valueSpearman = stats.spearmanr(GDD_TS, yield_TS)
    L_ = [Spearman, p_valueSpearman]
    GDD_spear.loc[GDD_spear["variety"]==a_variety, spearman_test_cols] = L_

GDD_spear["spearman"] = GDD_spear["spearman"].round(2)
GDD_spear["p_valSpearman"] = GDD_spear["p_valSpearman"].round(4)

GDD_spear.sort_values(by= ['variety'], inplace=True)
GDD_spear.reset_index(drop=True, inplace=True)

idx_max = GDD_spear["spearman"].idxmax()
print (GDD_spear.loc[idx_max, "spearman"], GDD_spear.loc[idx_max, "p_valSpearman"])
idx_min = GDD_spear["spearman"].idxmin()
print (GDD_spear.loc[idx_min, "spearman"], GDD_spear.loc[idx_min, "p_valSpearman"])

GDD_spear

# %% [markdown]
# ### dGDD Spearman test

# %%
yield_TS = separate_varieties_annual["yield"].values
dGDD_TS = separate_varieties_annual["year_dgdd"].values
Spearman, p_valueSpearman = stats.spearmanr(dGDD_TS, yield_TS)
print (Spearman.round(3), p_valueSpearman.round(7))

# %%
DF = separate_varieties_annual.copy() # make the lines shorter!
dGDD_spear = DF["variety"].copy()
dGDD_spear.drop_duplicates(inplace=True)

spearman_test_cols = ["spearman", "p_valSpearman"]
dGDD_spear = pd.concat([dGDD_spear, pd.DataFrame(columns = spearman_test_cols)])
dGDD_spear[spearman_test_cols] = [-666, -666] * (len(spearman_test_cols)-1)

for a_variety in dGDD_spear["variety"].unique():
    yield_TS = DF.loc[DF["variety"]==a_variety, "yield"].values
    GDD_TS = DF.loc[DF["variety"]==a_variety, "year_dgdd"].values
    Spearman, p_valueSpearman = stats.spearmanr(GDD_TS, yield_TS)
    L_ = [Spearman, p_valueSpearman]
    dGDD_spear.loc[dGDD_spear["variety"]==a_variety, spearman_test_cols] = L_

dGDD_spear["spearman"] = dGDD_spear["spearman"].round(2)
dGDD_spear["p_valSpearman"] = dGDD_spear["p_valSpearman"].round(4)

dGDD_spear.sort_values(by= ['variety'], inplace=True)
dGDD_spear.reset_index(drop=True, inplace=True)

idx_max = dGDD_spear["spearman"].idxmax()
print (dGDD_spear.loc[idx_max, "spearman"], dGDD_spear.loc[idx_max, "p_valSpearman"])

idx_min = dGDD_spear["spearman"].idxmin()
print (dGDD_spear.loc[idx_min, "spearman"], dGDD_spear.loc[idx_min, "p_valSpearman"])

dGDD_spear

# %% [markdown]
# ### precipitation Spearman test

# %%
yield_TS = separate_varieties_annual["yield"].values
precip_TS = separate_varieties_annual["year_precip"].values
Spearman, p_valueSpearman = stats.spearmanr(precip_TS, yield_TS)
print (Spearman.round(3), p_valueSpearman.round(7))

# %%
DF = separate_varieties_annual.copy() # make the lines shorter!
precip_spear = DF["variety"].copy()
precip_spear.drop_duplicates(inplace=True)

spearman_test_cols = ["spearman", "p_valSpearman"]
precip_spear = pd.concat([precip_spear, pd.DataFrame(columns = spearman_test_cols)])
precip_spear[spearman_test_cols] = [-666, -666] * (len(spearman_test_cols)-1)

for a_variety in precip_spear["variety"].unique():
    yield_TS = DF.loc[DF["variety"]==a_variety, "yield"].values
    precip_TS = DF.loc[DF["variety"]==a_variety, "year_precip"].values
    Spearman, p_valueSpearman = stats.spearmanr(precip_TS, yield_TS)
    L_ = [Spearman, p_valueSpearman]
    precip_spear.loc[precip_spear["variety"]==a_variety, spearman_test_cols] = L_

precip_spear["spearman"] = precip_spear["spearman"].round(2)
precip_spear["p_valSpearman"] = precip_spear["p_valSpearman"].round(5)

precip_spear.sort_values(by= ['variety'], inplace=True)
precip_spear.reset_index(drop=True, inplace=True)

idx_max = precip_spear["spearman"].idxmax()
print (precip_spear.loc[idx_max, "spearman"], precip_spear.loc[idx_max, "p_valSpearman"])

idx_min = precip_spear["spearman"].idxmin()
print (precip_spear.loc[idx_min, "spearman"], precip_spear.loc[idx_min, "p_valSpearman"])

precip_spear

# %% [markdown]
# ### Linear Regression with annual data

# %%
from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda

# %%
df_year = separate_varieties_annual.copy()

# %%
depen_var, indp_vars = "yield", ["year_precip"]

m5 = spreg.OLS_Regimes(y = np.log(df_year[depen_var].values), 
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

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out coeffs
                           "Std. Error": m5.std_err.flatten(),   
                           "P-Value": [i[1] for i in m5.t_stat],
                           }, index=m5.name_x)

print (f"{m5.r2.round(2) = }")
m5_results.transpose()

# %%
FontSize_ = 8
params = {"legend.fontsize": FontSize_*0.8,
          "legend.title_fontsize" : FontSize_ * 1.3,
          "legend.markerscale" : 1.5,
          "axes.labelsize": FontSize_ * 1,
          "axes.titlesize": FontSize_ * 1.5,
          "xtick.labelsize": FontSize_ * 1,
          "ytick.labelsize": FontSize_ * 1,
          "axes.titlepad": 10}
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(1, 1, figsize=(6, 2), sharex=True, 
                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);

title_ = f"log(yield) = $f(P)$"
axes.set_title(title_, fontsize=10);
axes.set_xlabel("prediction (in log form)"); axes.set_ylabel("residual");

fig_name = wheat_plot_dir + "logYield_PrecipReg.pdf"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

# %%

# %%
depen_var, indp_vars = "yield", ["all_gdd"]

m5 = spreg.OLS_Regimes(y = df_year[depen_var].values,  x = df_year[indp_vars].values, 
                       regimes = df_year["variety"].tolist(),
                       constant_regi="many",          
                       regime_err_sep=False,
                       name_y=depen_var,
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), 
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results.transpose()

# %%

# %%
depen_var, indp_vars = "yield", ["all_gdd", "all_precip"]

m5 = spreg.OLS_Regimes(y=df_year[depen_var].values, x=df_year[indp_vars].values, 
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
