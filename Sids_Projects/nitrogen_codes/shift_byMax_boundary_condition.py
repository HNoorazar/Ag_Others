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
import pandas as pd
import numpy as np

import scipy
import scipy.signal
import os, os.path
import datetime
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sb

import sys
import io

from pylab import rcParams

import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

# %% [markdown]
# ### Directories

# %%
data_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/"

# %% [markdown]
# ### Read Data

# %%
corn_potatoEq2_smoothed_daily = pd.read_csv(data_dir + "01_corn_potatoEq2_smoothed_daily.csv")

# %%
print (corn_potatoEq2_smoothed_daily.shape)
print('Number of fields is [{}].'.format(len(corn_potatoEq2_smoothed_daily.ID.unique())))
print('Number of OBJECTID is [{}].'.format(len(corn_potatoEq2_smoothed_daily.OBJECTID.unique())))

corn_potatoEq2_smoothed_daily.head(5)

# %% [markdown]
# ### Find the max and shift in a cycle fashion

# %%
desired_col = "smooth_window3"
field_IDs = corn_potatoEq2_smoothed_daily.ID.unique()
shifted_dt = corn_potatoEq2_smoothed_daily.copy()
shifted_dt = shifted_dt[["ID", "smooth_window3", "OBJECTID", "CropTyp", "Acres", "county"]]
# shifted_dt = pd.DataFrame()
# shifted_dt["ID"] = corn_potatoEq2_smoothed_daily.ID
# shifted_dt["OBJECTID"] = corn_potatoEq2_smoothed_daily.OBJECTID
# shifted_dt["smooth_window3"] = corn_potatoEq2_smoothed_daily.smooth_window3
# shifted_dt["CropTyp"] = corn_potatoEq2_smoothed_daily.CropTyp
# shifted_dt["Acres"] = corn_potatoEq2_smoothed_daily.Acres
# shifted_dt["county"] = corn_potatoEq2_smoothed_daily.county

shifted_dt.head(2)

# %%
for an_ID in field_IDs:
    curr_field = shifted_dt[shifted_dt.ID==an_ID]
    max_idx = curr_field.smooth_window3.idxmax()
    new_order = list(curr_field.loc[max_idx:, "smooth_window3"])+list(curr_field.loc[:max_idx-1, "smooth_window3"])
    shifted_dt.loc[curr_field.index, "smooth_window3"]=new_order

# %%
shifted_dt.head(2)

# %%
corn_potatoEq2_smoothed_daily.head(2)

# %%
an_ID = field_IDs[0]
fig, ax2 = plt.subplots(1, 1, figsize=(15, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

# ax2.plot(shifted_dt.loc[daily_DF[daily_DF.ID==a_field].nit.dropna().index, 'human_system_start_time'],
#          daily_DF[daily_DF.ID==a_field].nit.dropna(), 
#          linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");

ax2.plot(corn_potatoEq2_smoothed_daily[corn_potatoEq2_smoothed_daily.ID==an_ID].smooth_window3, 
         linewidth=3, ls='-', label='WMA-3', c="k");

ax2.plot(shifted_dt[shifted_dt.ID==an_ID].smooth_window3, 
         linewidth=3, ls='-', label='shifted WMA-3', c="dodgerblue");

raw_data = corn_potatoEq2_smoothed_daily[corn_potatoEq2_smoothed_daily.ID==an_ID].copy()
raw_data.dropna(subset = ["nit"], inplace=True)
ax2.plot(raw_data.nit, linewidth=3, ls='-', label='raw', c="c");

# ax2.plot(daily_DF[daily_DF.ID==a_field].smooth_window5, linewidth=3, ls='-', label='WMA-5', c="r");

# ax2.tick_params(axis='y', which='major') #, labelsize = tick_FontSize) #
# ax2.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
# ax2.xaxis.set_major_locator(mdates.MonthLocator())
# ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(False);
# ax2.set_xticklabels([]);

# %%

raw_data.head(2)

# %%
out_name = data_dir + "02_shifted_corn_potatoEq2_smoothed.csv"
shifted_dt.to_csv(out_name, index = False)

# %% [markdown]
# ### Scale each field between 0 and 1!

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
scaled_dt =shifted_dt.copy()

# %%
for an_ID in field_IDs:
    curr_field = scaled_dt[scaled_dt.ID==an_ID]
    scaler = MinMaxScaler()
    aa = scaler.fit_transform(curr_field[['smooth_window3']]).reshape(-1)
    scaled_dt.loc[curr_field.index, "smooth_window3"]=aa

# %%
an_ID = field_IDs[0]
fig, ax2 = plt.subplots(1, 1, figsize=(15, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

# ax2.plot(shifted_dt.loc[daily_DF[daily_DF.ID==a_field].nit.dropna().index, 'human_system_start_time'],
#          daily_DF[daily_DF.ID==a_field].nit.dropna(), 
#          linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");

ax2.plot(corn_potatoEq2_smoothed_daily[corn_potatoEq2_smoothed_daily.ID==an_ID].smooth_window3, 
         linewidth=3, ls='-', label='WMA-3', c="k");

ax2.plot(shifted_dt[shifted_dt.ID==an_ID].smooth_window3, 
         linewidth=3, ls='-', label='shifted WMA-3', c="dodgerblue");

ax2.plot(scaled_dt[scaled_dt.ID==an_ID].smooth_window3, 
         linewidth=3, ls='-', label='scaled - shifted WMA-3', c="r");

raw_data = corn_potatoEq2_smoothed_daily[corn_potatoEq2_smoothed_daily.ID==an_ID].copy()
raw_data.dropna(subset = ["nit"], inplace=True)
# ax2.plot(raw_data.nit, linewidth=3, ls='-', label='raw', c="c");

ax2.set_ylim(-0.1, 2)
# ax2.plot(daily_DF[daily_DF.ID==a_field].smooth_window5, linewidth=3, ls='-', label='WMA-5', c="r");

# ax2.tick_params(axis='y', which='major') #, labelsize = tick_FontSize) #
# ax2.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
# ax2.xaxis.set_major_locator(mdates.MonthLocator())
# ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(False);
# ax2.set_xticklabels([]);

# %%
out_name = data_dir + "03_scaled_shifted_corn_potatoEq2_smoothed.csv"
scaled_dt.to_csv(out_name, index = False)

# %%
