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
import scipy, scipy.signal

from datetime import date
import time

from random import seed
from random import random
import random
import os, os.path
import shutil

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import h5py
import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %%
data_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/"

fName="Corn_Potato_Sent_2020-01-01_2021-01-01.csv"
potatoCornRed = pd.read_csv(data_dir + fName)

meta=pd.read_csv(data_dir + "corn_potato_metadata2020.csv")

# %%
potatoCornRed.head(2)

# %%
meta.head(2)

# %%
potatoCornRed = pd.merge(potatoCornRed, meta[["ID", "ExactAcres"]], on=['ID'], how='left')
print (len(potatoCornRed.ID.unique()))
potatoCornRed=potatoCornRed[potatoCornRed.ExactAcres>10].copy()
print (len(potatoCornRed.ID.unique()))

# %%
potatoCornRed.dropna(subset=['CIRed'], inplace=True)

# %%
potatoCornRed.CropTyp.unique()

# %%
potato = potatoCornRed[potatoCornRed.CropTyp.isin(["Potato Seed", "Potato"])].copy()
corn = potatoCornRed[potatoCornRed.CropTyp.isin(['Corn, Field', 'Corn, Sweet', 'Corn Seed'])].copy()

potato.reset_index(drop=True, inplace=True)
corn.reset_index(drop=True, inplace=True)

# %%
corn["chl"] =corn["CIRed"]* 6.68
corn["chl"] =corn["chl"]-0.67

potato.loc[:, 'chl'] = potato.loc[:, 'CIRed']*0.8013
potato.loc[:, 'chl'] = potato.loc[:, 'chl']-0.4704

corn_potato = pd.concat([corn, potato])
corn_potato.reset_index(drop=True, inplace=True)

# %%
corn_potato["nit"] = corn_potato["chl"]*4.73+0.27
corn_potato.head(2)

# %%
corn_potato = nc.add_human_start_time_by_system_start_time(corn_potato)
corn_potato.head(2)

# %%
###### Round the damn numbers
corn_potato=corn_potato.round(3)

# %%

# %%
potato_nit_min=corn_potato[corn_potato.CropTyp.isin(["Potato Seed", "Potato"])].nit.min()
potato_nit_max=corn_potato[corn_potato.CropTyp.isin(["Potato Seed", "Potato"])].nit.max()

corn_nit_min=corn_potato[~corn_potato.CropTyp.isin(["Potato Seed", "Potato"])].nit.min()
corn_nit_max=corn_potato[~corn_potato.CropTyp.isin(["Potato Seed", "Potato"])].nit.max()


IDs=corn_potato.ID.unique()

# %%
_id=IDs[0]


# %%
def plot_oneColumn_CropTitle_scatter(raw_dt, ax, titlee, idx="NDVI", 
                                     _label = "raw", _color="red", 
                                     y_min=-1, y_max=1):

    ax.plot(raw_dt['human_system_start_time'], raw_dt[idx], c=_color, linewidth=2,
                label=_label);

    ax.set_title(titlee)
    ax.set_ylabel(idx, fontsize=20) # , labelpad=20); # fontsize = label_FontSize,
    ax.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    ax.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
    ax.legend(loc="upper right");
    # plt.yticks(np.arange(0, 1, 0.2))
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.set_ylim(y_min-0.1, y_max+0.1)


# %%
size = 20
title_FontSize = 10
legend_FontSize = 14
tick_FontSize = 18
label_FontSize = 14

params = {'legend.fontsize': 17,
          'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.75,
          'ytick.labelsize': size * 0.75,
          'axes.titlepad': 10}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)

def plot_oneColumn_CropTitle_scatter(raw_dt, ax, titlee, idx="NDVI", 
                                     _label = "raw", _color="red", 
                                     _marker_shape="+",
                                     marker_size=60, y_min=-1, y_max=1):

    ax.plot(raw_dt['human_system_start_time'], raw_dt[idx], c=_color, linewidth=2,
                label=_label);

    ax.set_title(titlee)
    ax.set_ylabel(idx, fontsize=20) # , labelpad=20); # fontsize = label_FontSize,
    ax.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    ax.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
    ax.legend(loc="upper right");
    # plt.yticks(np.arange(0, 1, 0.2))
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.set_ylim(y_min-0.1, y_max+0.1)

    


indeks="nit"

curr = corn_potato[corn_potato.ID == _id].copy()
curr.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
curr_meta = meta[meta.ID==_id]
titlee = " ".join(curr_meta.CropTyp.unique()[0].split(", ")[::-1])

fig, axs = plt.subplots(1, 1, figsize=(15, 4), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(True);
plot_oneColumn_CropTitle_scatter(raw_dt = curr, ax=axs, idx=indeks, titlee=titlee,
                                 _label = "Canopy N", marker_size=40,
                                 _color="dodgerblue",
                                y_min=corn_nit_min, y_max=corn_nit_max)

# %%

# %%
A = corn_potato[corn_potato.CropTyp=='Corn, Field']
print (A.nit.min())
print (A.nit.max())

# %%
A = corn_potato[corn_potato.CropTyp=='Corn, Sweet']
print (A.nit.min())
print (A.nit.max())

# %%
A = corn_potato[corn_potato.CropTyp=='Corn Seed']
print (A.nit.min())
print (A.nit.max())

# %%
A = corn_potato[corn_potato.CropTyp=='Potato']
print (A.nit.min())
print (A.nit.max())

# %%
A = corn_potato[corn_potato.CropTyp=='Potato Seed']
print (A.nit.min())
print (A.nit.max())

# %%
print (sorted(corn_potato[corn_potato.CropTyp=='Corn, Field'].nit)[::-1][:10])

# %%
field_corn = corn_potato[corn_potato.CropTyp=='Corn, Field'].copy()
field_corn.sort_values(by=['nit'], inplace=True, ascending=False)
field_corn.reset_index(drop=True, inplace=True)

meta[meta.ID==field_corn.ID[0]]

# %%
corn_potato.groupby(['CropTyp'])['ID'].nunique()

# %%
corn_potato[corn_potato.CropTyp=="Potato Seed"].ID.unique()

# %%
potato = corn_potato[corn_potato.CropTyp=='Potato'].copy()
potato.sort_values(by=['nit'], inplace=True, ascending=False)
potato.reset_index(drop=True, inplace=True)

meta[meta.ID==potato.ID[0]]

# %%

# %% [markdown]
# # Eq. 2 for Potato

# %%
data_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/"

fName="Potato_Sent_Sept28_2ndFormula_2020-01-01_2021-01-01.csv"
potato = pd.read_csv(data_dir + fName)

meta=pd.read_csv(data_dir + "corn_potato_metadata2020.csv")

# %%
potato["nit"] = potato["CIRed"]*0.66+1.82
potato.head(2)

# %%
a_field=potato[potato.ID=="82131_WSDA_SF_2020"].copy()

a_field_nit=a_field.copy()
a_field_GEEnit=a_field.copy()

a_field_GEEnit.dropna(subset=['NuptakeGEE'], inplace=True)
a_field_nit.dropna(subset=['nit'], inplace=True)

# %%
print (a_field_nit.shape)
print (a_field_GEEnit.shape)

# %%
a_field_GEEnit = nc.add_human_start_time_by_system_start_time(a_field_GEEnit)
a_field_nit = nc.add_human_start_time_by_system_start_time(a_field_nit)

# %%
size = 20; title_FontSize = 10; legend_FontSize = 14
tick_FontSize = 18; label_FontSize = 14

params = {'legend.fontsize': 17,
          'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.75,
          'ytick.labelsize': size * 0.75,
          'axes.titlepad': 10}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)

fig, ax = plt.subplots(1, 1, figsize=(15, 4), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
ax.grid(True);

ax.plot(a_field_nit['human_system_start_time'], a_field_nit["nit"], c="dodgerblue", linewidth=10,
        label="nit");

ax.plot(a_field_GEEnit['human_system_start_time'], a_field_GEEnit["NuptakeGEE"], c="red", linewidth=4,
        label="nit");

ax.set_ylabel("nitrogen", fontsize=20) # , labelpad=20); # fontsize = label_FontSize,
ax.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
ax.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax.legend(loc="upper right");
# plt.yticks(np.arange(0, 1, 0.2))
# ax.xaxis.set_major_locator(mdates.YearLocator(1))
# ax.set_ylim(y_min-0.1, y_max+0.1)


# %%

# %%

# %%
