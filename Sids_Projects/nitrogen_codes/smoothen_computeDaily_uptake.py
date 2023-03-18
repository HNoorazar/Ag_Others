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
# # Dec 1.
#
# Meeting with Kirti and Sid.
#
# We want to smoothen the curves and then compute daily uptakes.
#    - Get rid of small fields.
#    - Smoothen.
#    - daily uptake.
#    - Clip to Apr - Oct (inclusive)
#    - Standardize and cloud-plots (to show range) for later.

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
# ### Set up directories

# %%
data_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/"

# %%
##
## Read data
##
potato = pd.read_csv(data_dir + "Potato_Sent_Sept28_2ndFormula_2020-01-01_2021-01-01.csv")
corn = pd.read_csv(data_dir + "Corn_Potato_Sent_2020-01-01_2021-01-01.csv")
metadata = pd.read_csv(data_dir + "corn_potato_metadata2020.csv")

# %%
print('We have [{}] fields in total.'.format(len(metadata.ID.unique())))
print('There are [{}] fields that are smaller than 10 acres.'.format(metadata[metadata.ExactAcres<=10].shape[0]))

# %%
## Toss small fields
metadata=metadata[metadata.ExactAcres>10]

# %% [markdown]
# #### We are only interested in "field corn" and "potato". In other words no seeds and no sweet corn.

# %%
metadata.CropTyp.unique()

# %%
print (metadata.shape)
metadata=metadata[metadata.CropTyp.isin(["Corn, Field", "Potato"])]
print (metadata.shape)

# %%
metadata.head(2)

# %%
potato.head(2)

# %% [markdown]
# ### Corn Uptake
# Uptake for potato was computed on GEE but for corn we did it locally.
# The reason was that their equation was different. It was easier to do locally.
# Later we found a better equation for potato and I did it on GEE. So, here we need to compute the uptake
# for corn.

# %%
##
##   Pick up only big fields and drop the seeds.
##

print('There are [{}] fields of potato.'.format(len(potato.ID.unique())))
potato = potato[potato.ID.isin(list(metadata.ID))]
print('There are [{}] large fields of potato.'.format(len(potato.ID.unique())))

print ("================================================================================")
print ("")
print ("The crop type in here --> {} must be potato, is it?".format(potato.CropTyp.unique()))
print ("")
print ("================================================================================")

print('There are [{}] fields of corn.'.format(len(corn.ID.unique())))
corn = corn[corn.ID.isin(list(metadata.ID))]
print('There are [{}] large fields of "corn field".'.format(len(corn.ID.unique())))

# %%
# Drop NAs in potato
print (potato.shape)
potato.dropna(subset=["NuptakeGEE"], axis=0, inplace=True)
print (potato.shape)

potato.reset_index(drop=True, inplace=True)
potato.head(2)

# %% [markdown]
# ### Compute Corn's uptake

# %%
print (corn.shape)
corn=corn[corn.CropTyp.isin(["Corn, Field"])]
print (corn.shape)
print ("================================================================================")
print ("")
print ("The crop type in here --> {} must be [corn field], is it?".format(corn.CropTyp.unique()))
print ("")
print ("================================================================================")

print (corn.shape)
corn.dropna(subset=["CIRed"], axis=0, inplace=True)
print (corn.shape)

corn.reset_index(drop=True, inplace=True)

# %%
corn["chl"]=corn["CIRed"]*6.68
corn["chl"]=corn["chl"]-0.67
corn["nit"] = corn["chl"]*4.73+0.27

corn.head(3)

# %% [markdown]
# #### concatenate the two dataframes for ease of use

# %%
potato.rename(columns={'NuptakeGEE': 'nit'}, inplace=True)

# %%
corn_potato = pd.concat([
                         potato[["ID", "CropTyp", "system_start_time", "nit"]],
                         corn[["ID", "CropTyp", "system_start_time", "nit"]]
                         ], 
                         ignore_index=True)

corn_potato.sort_values(by=["ID", "system_start_time"], inplace=True)
corn_potato.reset_index(drop=True, inplace=True)
corn_potato.head(2)

# %%
corn_potato.groupby('CropTyp').agg('min')

# %%
corn_potato.groupby('CropTyp').agg('max')

# %% [markdown]
# #### Smoothen with weighted moving average for windows of size 3 and 5

# %%
windows=[3, 5]

for window_ in windows:
    new_col = "smooth_window" + str(window_)
    corn_potato[new_col] = corn_potato.nit
    for a_field in corn_potato.ID.unique():
        a_signal = corn_potato[corn_potato.ID==a_field][new_col]
        weights = np.arange(1, window_+1)
        wma = a_signal.rolling(window_).apply(lambda a_signal: np.dot(a_signal, weights)/weights.sum(), raw=True)
        corn_potato.loc[wma.index, new_col] = wma


# %%
corn_potato.groupby('CropTyp').agg('min')

# %%
corn_potato.groupby('CropTyp').agg('max')

# %%
# window = 5
# a_signal = corn_potato[corn_potato.ID==a_field].nit
# weights = np.arange(1, window+1)
# wma = a_signal.rolling(window).apply(lambda a_signal: np.dot(a_signal, weights)/weights.sum(), raw=True)
# corn_potato.loc[wma.index, "smooth_window5"] = wma
# corn_potato[corn_potato.ID==a_field]

# %%
corn_potato.head(2)

# %% [markdown]
# ## which window_ is better one?

# %%
import time
def add_human_start_time_by_system_start_time(HDF):
    """Returns human readable time (conversion of system_start_time)

    Arguments
    ---------
    HDF : dataframe

    Returns
    -------
    HDF : dataframe
        the same dataframe with added column of human readable time.
    """
    HDF.system_start_time = HDF.system_start_time / 1000
    time_array = HDF["system_start_time"].values.copy()
    human_time_array = [time.strftime('%Y-%m-%d', time.localtime(x)) for x in time_array]
    HDF["human_system_start_time"] = human_time_array

    if type(HDF["human_system_start_time"]==str):
        HDF['human_system_start_time'] = pd.to_datetime(HDF['human_system_start_time'])
    
    """
    Lets do this to go back to the original number:
    I added this when I was working on Colab on March 30, 2022.
    Keep an eye on it and see if we have ever used "system_start_time"
    again. If we do, how we use it; i.e. do we need to get rid of the 
    following line or not.
    """
    HDF.system_start_time = HDF.system_start_time * 1000
    return(HDF)


# %%
corn_potato = add_human_start_time_by_system_start_time(corn_potato)
corn_potato.head(2)

# %%
corn_potato['human_system_start_time'] = pd.to_datetime(corn_potato['human_system_start_time'])

# %%
fig, ax2 = plt.subplots(1, 1, figsize=(10, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time, 
         corn_potato[corn_potato.ID==a_field].nit, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time, 
         corn_potato[corn_potato.ID==a_field].smooth_window3, linewidth=3, ls='-', label='WMA-3', c="k");
ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time, 
         corn_potato[corn_potato.ID==a_field].smooth_window5, linewidth=3, ls='-', label='WMA-5', c="r");

ax2.tick_params(axis='y', which='major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

plot_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/report_plots/"
os.makedirs(plot_dir, exist_ok=True)
file_name = plot_dir + "wma.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + "wma.png"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%
fig, ax2 = plt.subplots(1, 1, figsize=(10, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time, 
         corn_potato[corn_potato.ID==a_field].nit, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time, 
         corn_potato[corn_potato.ID==a_field].smooth_window3, linewidth=3, ls='-', label='WMA-3', c="k");
# ax2.plot(corn_potato[corn_potato.ID==a_field].smooth_window5, linewidth=3, ls='-', label='WMA-5', c="r");

ax2.tick_params(axis='y', which='major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

plot_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/report_plots/"
os.makedirs(plot_dir, exist_ok=True)
file_name = plot_dir + "wma_no5.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + "wma_no5.png"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);


# %%
a_field=corn_potato.ID.unique()[0]
# a_field="91673_WSDA_SF_2020"
fig, ax2 = plt.subplots(1, 1, figsize=(10, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time, 
         corn_potato[corn_potato.ID==a_field].nit, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time, 
         corn_potato[corn_potato.ID==a_field].smooth_window3, linewidth=3, ls='-', label='WMA-3', c="k");
ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time, 
         corn_potato[corn_potato.ID==a_field].smooth_window5, linewidth=3, ls='-', label='WMA-5', c="r");

ax2.tick_params(axis='y', which='major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

# %%
"52036_WSDA_SF_2020" in list(corn_potato.ID.unique())

# %%
corn_potato.head(2)

# %%
out_name = data_dir + "corn_potatoEq2_smoothed.csv"
corn_potato.to_csv(out_name, index = False)

# %% [markdown]
# # Do a daily full year interpolation

# %% [markdown]
# ### Create a daily DataFrame!

# %%
field_IDs = corn_potato.ID.unique()
number_of_fields = len(field_IDs)


base = pd.to_datetime("2020-01-1")
full_year_date_list = [base + datetime.timedelta(days=x) for x in range(365)]

field_IDs_repeated_4_full_year = np.repeat(field_IDs, 365)
calendar_repeated_4_full_year = np.tile(full_year_date_list, number_of_fields)

daily_DF = pd.DataFrame()
daily_DF["ID"] = field_IDs_repeated_4_full_year
daily_DF["human_system_start_time"] = calendar_repeated_4_full_year

daily_DF = pd.merge(daily_DF, 
                    corn_potato[["ID", "human_system_start_time", "nit", "smooth_window3", "smooth_window5"]],
                    on=['ID', 'human_system_start_time'], how='outer')

daily_DF["DoY"] = daily_DF.human_system_start_time.dt.dayofyear 
daily_DF.head(14)

# %%
corn_potato.CropTyp.unique()

# %%
len(corn_potato.ID.unique())

# %%
2407*365

# %%
len(field_IDs_repeated_4_full_year)

# %%
daily_DF.head(10)

# %%
corn_potato.shape

# %%
daily_DF.shape

# %% [markdown]
# ### Start and End of the year
#  For some fields, such as ```100457_WSDA_SF_2020``` the first data we have is for Feb. 2.
#  I replace Jan 1 - Feb. 2 with the same value as Feb. 2. Interpolation is not doable!
#  I can extrapolate I guess! That would not be good tho!
#  So, lets take of beginning and end of the year and fill those in!

# %%
import scipy.interpolate

for an_ID in daily_DF.ID.unique():
    curr_DF_subset = daily_DF[daily_DF.ID==an_ID]
    
    slice_idx_min = curr_DF_subset.index[0]
    slice_idx_max = curr_DF_subset.index[-1]

    # find index of firs NA and last NA and fill whatever NA
    # that there is before and after them
    
    ####
    #### nit (do not fill the gap in the original nit's. Just do the smoothed versions!)
    ####
#     first_notNA_idx = curr_DF_subset.nit.first_valid_index()
#     last_notNA_idx  = curr_DF_subset.nit.last_valid_index()
    
#     first_notNA_value = curr_DF_subset.loc[first_notNA_idx]["nit"]
#     last_notNA_value  = curr_DF_subset.loc[last_notNA_idx]["nit"]
    
#     ### Replace potential NAs at the beginning and end:
#     daily_DF.loc[slice_idx_min:first_notNA_idx, "nit"] = first_notNA_value
#     daily_DF.loc[last_notNA_idx:slice_idx_max, "nit"] = last_notNA_value
    
    ####
    #### smooth_window3
    ####
    first_notNA_idx = curr_DF_subset.smooth_window3.first_valid_index()
    last_notNA_idx  = curr_DF_subset.smooth_window3.last_valid_index()
    
    first_notNA_value = curr_DF_subset.loc[first_notNA_idx]["smooth_window3"]
    last_notNA_value  = curr_DF_subset.loc[last_notNA_idx]["smooth_window3"]
    
    ### Replace potential NAs at the beginning and end:
    daily_DF.loc[slice_idx_min:first_notNA_idx, "smooth_window3"] = first_notNA_value
    daily_DF.loc[last_notNA_idx:slice_idx_max, "smooth_window3"] = last_notNA_value
    
    ####
    #### smooth_window5
    ####
    first_notNA_idx = curr_DF_subset.smooth_window5.first_valid_index()
    last_notNA_idx  = curr_DF_subset.smooth_window5.last_valid_index()
    
    first_notNA_value = curr_DF_subset.loc[first_notNA_idx]["smooth_window5"]
    last_notNA_value  = curr_DF_subset.loc[last_notNA_idx]["smooth_window5"]
    
    ### Replace potential NAs at the beginning and end:
    daily_DF.loc[slice_idx_min:first_notNA_idx, "smooth_window5"] = first_notNA_value
    daily_DF.loc[last_notNA_idx:slice_idx_max, "smooth_window5"] = last_notNA_value
    
    ####
    ####   interpolate between the gaps now
    ####
    # redo the following line so we have an updated version!
    curr_DF_subset = daily_DF[daily_DF.ID==an_ID]
    
    ###
    ###    interpolate smooth_window3
    ###
    # subset non-NAs for interpolation
    not_NA_idx = curr_DF_subset[curr_DF_subset['smooth_window3'].notnull()]
    not_NA_idx = not_NA_idx.index
    
    # interpolate equations
    x = curr_DF_subset.loc[not_NA_idx, "DoY"].values
    y = curr_DF_subset.loc[not_NA_idx, "smooth_window3"].values
    y_interp_model = scipy.interpolate.interp1d(x, y)
    
    # interpolate everything
    y_interps = y_interp_model(curr_DF_subset.DoY.values)
    daily_DF.loc[curr_DF_subset.index, "smooth_window3"] = y_interps
    
    ###
    ###    interpolate smooth_window5
    ###
    # subset non-NAs for interpolation
    not_NA_idx = curr_DF_subset[curr_DF_subset['smooth_window3'].notnull()]
    not_NA_idx = not_NA_idx.index
    
    # subset non-NAs for interpolation
    not_NA_idx = curr_DF_subset[curr_DF_subset['smooth_window5'].notnull()]
    not_NA_idx = not_NA_idx.index
    
    # interpolate equations
    x = curr_DF_subset.loc[not_NA_idx, "DoY"].values
    y = curr_DF_subset.loc[not_NA_idx, "smooth_window5"].values
    y_interp_model = scipy.interpolate.interp1d(x, y)
    
    # interpolate everything
    y_interps = y_interp_model(curr_DF_subset.DoY.values)
    daily_DF.loc[curr_DF_subset.index, "smooth_window5"] = y_interps

# %%
print (daily_DF.shape)
daily_DF.dropna(subset=["smooth_window3"]).shape

# %%
daily_DF.head(2)

# %%
metadata.head(2)

# %%
## add metadata back to the dataframe!
daily_DF = pd.merge(daily_DF, 
                    metadata[["ID", "OBJECTID", "CropTyp", "Acres", "county"]], 
                    on=['ID'], how='left')

daily_DF.head(2)

# %%
out_name = data_dir + "01_corn_potatoEq2_smoothed_daily.csv"
daily_DF.to_csv(out_name, index = False)

# %%
fig, ax2 = plt.subplots(1, 1, figsize=(15, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(daily_DF.loc[daily_DF[daily_DF.ID==a_field].nit.dropna().index, 'human_system_start_time'],
         daily_DF[daily_DF.ID==a_field].nit.dropna(), 
         linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");

ax2.scatter(daily_DF[daily_DF.ID==a_field].human_system_start_time, 
            daily_DF[daily_DF.ID==a_field].smooth_window3, 
            marker='+', s=124, c='r', label="WMA-3 scatter")

ax2.plot(daily_DF[daily_DF.ID==a_field].human_system_start_time,
         daily_DF[daily_DF.ID==a_field].smooth_window3, linewidth=5, ls='-', label='WMA-3', c="k");


# ax2.plot(daily_DF[daily_DF.ID==a_field].smooth_window5, linewidth=3, ls='-', label='WMA-5', c="r");

ax2.tick_params(axis='y', which='major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
# ax2.xaxis.set_major_locator(mdates.MonthLocator())
# ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

# %%
VI_idx = "V"
smooth_type = "N"
SR = 3
print (f"Passed Args. are: {VI_idx=:}, {smooth_type=:}, and {SR=:}!")

# %%

# %%
