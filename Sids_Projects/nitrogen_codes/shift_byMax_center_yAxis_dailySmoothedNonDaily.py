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

# %%
corn_potatoEq2_smoothed_daily = pd.read_csv(data_dir + "01_corn_potatoEq2_smoothed_daily.csv")
corn_potatoEq2_smoothed_daily.head(2)

# %% [markdown]
# ### replace the daily values that were NA with NA again, so that the cloud plots make sense

# %%
nit_NA_indices = np.where(corn_potatoEq2_smoothed_daily['nit'].isna())[0]

# %%
corn_potatoEq2_smoothed_daily.loc[list(nit_NA_indices), "smooth_window3"] = float("nan")

# %% [markdown]
# ### Read Data

# %%
print (corn_potatoEq2_smoothed_daily.shape)
print('Number of fields is [{}].'.format(len(corn_potatoEq2_smoothed_daily.ID.unique())))
corn_potatoEq2_smoothed_daily.head(5)

# %%
A = np.where(corn_potatoEq2_smoothed_daily['nit'].isna())[0]
B = np.where(corn_potatoEq2_smoothed_daily['smooth_window3'].isna())[0]

sum(A==B)==len(A)

# %% [markdown]
# ### Find the max and center on y-axis

# %%
desired_col = "smooth_window3"
field_IDs = corn_potatoEq2_smoothed_daily.ID.unique()
shifted_dt = corn_potatoEq2_smoothed_daily.copy()
shifted_dt = shifted_dt[["ID", "smooth_window3", "OBJECTID", "CropTyp", "Acres", "county", "DoY"]]
# shifted_dt = pd.DataFrame()
# shifted_dt["ID"] = corn_potatoEq2_smoothed_daily.ID
# shifted_dt["OBJECTID"] = corn_potatoEq2_smoothed_daily.OBJECTID
# shifted_dt["smooth_window3"] = corn_potatoEq2_smoothed_daily.smooth_window3
# shifted_dt["CropTyp"] = corn_potatoEq2_smoothed_daily.CropTyp
# shifted_dt["Acres"] = corn_potatoEq2_smoothed_daily.Acres
# shifted_dt["county"] = corn_potatoEq2_smoothed_daily.county
shifted_dt["x_axis"] = 0
shifted_dt.head(2)

# %%
for an_ID in field_IDs:
    curr_field = shifted_dt[shifted_dt.ID==an_ID]
    max_idx = curr_field.smooth_window3.idxmax()
    curr_doy = curr_field.loc[max_idx, "DoY"]
    # new_order = list(-1*np.arange(0, curr_doy)[::-1])+list(np.arange(1, 365-curr_doy+1))
    new_order = curr_field.DoY-curr_doy
    shifted_dt.loc[curr_field.index, "x_axis"]=new_order

# %%
shifted_dt.head(2)

# %%
shifted_dt.smooth_window3.unique()

# %%
corn_potatoEq2_smoothed_daily.head(2)

# %%
all_NA_in_shifted_dt_ID="100457_WSDA_SF_2020"
shifted_dt[shifted_dt.ID==an_ID].smooth_window3.unique()

# %%
corn_potatoEq2_smoothed_daily[corn_potatoEq2_smoothed_daily.ID==an_ID].smooth_window3.unique()

# %%
corn_potatoEq2_smoothed_daily[corn_potatoEq2_smoothed_daily.ID==an_ID]

# %%
an_ID = field_IDs[0]
fig, ax2 = plt.subplots(1, 1, figsize=(15, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

curr_smoothed_data = corn_potatoEq2_smoothed_daily[corn_potatoEq2_smoothed_daily.ID==an_ID].copy()
curr_smoothed_data.dropna(subset = ["nit"], inplace=True)
ax2.plot(curr_smoothed_data.DoY,
         curr_smoothed_data.smooth_window3, 
         linewidth=3, ls='-', label='WMA-3', c="k");

curr_shifted_data = shifted_dt[shifted_dt.ID==an_ID].copy()
curr_shifted_data.dropna(subset = ["smooth_window3"], inplace=True)
ax2.plot(curr_shifted_data.x_axis,
         curr_shifted_data.smooth_window3,
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
shifted_dt.dropna(subset = ["smooth_window3"], inplace=True)
out_name = data_dir + "02_yAxis_corn_potatoEq2_smoothed_nonDaily.csv"
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

ax2.plot(shifted_dt[shifted_dt.ID==an_ID].x_axis,
         shifted_dt[shifted_dt.ID==an_ID].smooth_window3,
         linewidth=3, ls='-', label='shifted WMA-3', c="dodgerblue");

ax2.plot(scaled_dt[scaled_dt.ID==an_ID].x_axis,
         scaled_dt[scaled_dt.ID==an_ID].smooth_window3, 
         linewidth=3, ls='-', label='scaled - shifted WMA-3', c="r");

raw_data = corn_potatoEq2_smoothed_daily[corn_potatoEq2_smoothed_daily.ID==an_ID].copy()
raw_data.dropna(subset = ["nit"], inplace=True)
# ax2.plot(raw_data.nit, linewidth=3, ls='-', label='raw', c="c");

# ax2.set_ylim(-0.1, 2)
# ax2.plot(daily_DF[daily_DF.ID==a_field].smooth_window5, linewidth=3, ls='-', label='WMA-5', c="r");

# ax2.tick_params(axis='y', which='major') #, labelsize = tick_FontSize) #
# ax2.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
# ax2.xaxis.set_major_locator(mdates.MonthLocator())
# ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(False);
# ax2.set_xticklabels([]);

# %%
scaled_dt.smooth_window3.unique()
scaled_dt.head(2)

# %%
out_name = data_dir + "03_scaled_yAxis_corn_potatoEq2_smoothed_nonDaily.csv"
scaled_dt.to_csv(out_name, index = False)

# %%
VI_idx = "V"
smooth_type = "N"
SR = 3
print (f"Passed Args. are: {VI_idx=:}, {smooth_type=:}, and {SR=:}!")

# %%
