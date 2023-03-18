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
# Here I am trying to look at LSWI, BSI, and NDSI and see if they work in our area

# %%
# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from math import factorial
import scipy
import scipy.signal
import os, os.path
from datetime import timedelta
import datetime
# from pprint import pprint
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sb

import sys
import io

from pylab import rcParams
# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

# from numpy.fft import rfft, irfft, rfftfreq, ifft
# from scipy import fft, fftpack
import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp

# %%
snow_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/"
PMW_diff_dir = snow_dir + "00/PMW_difference_data/"
SNOTEL_dir = snow_dir + "00/SNOTEL_data/"
GEE_data_dir = snow_dir + "01_GEE_data/"

# %%
SNOTEL_Snow_depth_2013 = pd.read_csv(SNOTEL_dir + "SNOTEL_Snow_depth_2013.csv")
SNOTEL_Snow_depth_2013['Date'] = pd.to_datetime(SNOTEL_Snow_depth_2013['Date'])

# %%
DayRule28_no_snow_first_DoY=pd.read_csv(snow_dir + "00/DayRule28_no_snow_first_DoY.csv")
DayRule28_no_snow_first_DoY['no_snow_first_DoY'] = pd.to_datetime(DayRule28_no_snow_first_DoY['no_snow_first_DoY'])
DayRule28_no_snow_first_DoY.head(2)

# %%
PMW = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013.csv")

# %%
NDSI_A2_L7 = pd.read_csv(GEE_data_dir+"NDSI_A2_L7.csv")
NDSI_A2_L8 = pd.read_csv(GEE_data_dir+"NDSI_A2_L8.csv")
indices = pd.concat([NDSI_A2_L7, NDSI_A2_L8])
indices.head(2)

# %%
BSI = indices[["Station_Name", "BSI", "system_start_time"]].copy()
NDSI = indices[["Station_Name", "NDSI", "system_start_time"]].copy()
LSWI = indices[["Station_Name", "LSWI", "system_start_time"]].copy()

# %%
# Drop NAs
BSI = BSI[BSI['BSI'].notna()]
NDSI = NDSI[NDSI['NDSI'].notna()]
LSWI = LSWI[LSWI['LSWI'].notna()]

# %%
BSI.head(2)

# %% [markdown]
# ### add human time

# %%
BSI = nc.add_human_start_time_by_system_start_time(BSI)
NDSI = nc.add_human_start_time_by_system_start_time(NDSI)
LSWI = nc.add_human_start_time_by_system_start_time(LSWI)

# %% [markdown]
# ### sort

# %%
BSI.sort_values(by=['Station_Name', 'human_system_start_time'], inplace=True)
NDSI.sort_values(by=['Station_Name', 'human_system_start_time'], inplace=True)
LSWI.sort_values(by=['Station_Name', 'human_system_start_time'], inplace=True)

BSI.reset_index(drop=True, inplace=True)
NDSI.reset_index(drop=True, inplace=True)
LSWI.reset_index(drop=True, inplace=True)

# %%
a_station = DayRule28_no_snow_first_DoY.Station_Name[0]

fig, ax2 = plt.subplots(1, 1, figsize=(15, 5), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});


ax2.plot(BSI.loc[BSI.Station_Name==a_station, "human_system_start_time"], 
         BSI.loc[BSI.Station_Name==a_station, "BSI"], 
         linewidth = 2, ls = '-', label = "BSI", c="k");

ax2.plot(NDSI.loc[NDSI.Station_Name==a_station, "human_system_start_time"], 
         NDSI.loc[NDSI.Station_Name==a_station, "NDSI"], 
         linewidth = 2, ls = '-', label = "NDSI", c="r");

ax2.plot(LSWI.loc[LSWI.Station_Name==a_station, "human_system_start_time"], 
         LSWI.loc[LSWI.Station_Name==a_station, "LSWI"], 
         linewidth = 2, ls = '-', label = "LSWI", c="g");

ax2.plot(SNOTEL_Snow_depth_2013['Date'], 
         SNOTEL_Snow_depth_2013[a_station], 
         linewidth = 2, ls = '-', label = "SNOTEL", c="dodgerblue");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) #
# ax2.xaxis.set_major_locator(mdates.MonthLocator())
# ax2.xaxis.set_major_formatter(DateFormatter('%b'))
plt.ylim([-2, 4])

ax2.legend(loc="upper left");
ax2.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

# plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
# plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# %%
station_names = list(BSI["Station_Name"].unique())
print (len(station_names))
print (station_names[:4])

import random
random.seed(0)
random.shuffle(station_names)

station_count_2_plot=10
fig, axs = plt.subplots(station_count_2_plot, 1, 
                        figsize=(10, station_count_2_plot*3),
                        gridspec_kw={'hspace': 0.5, 'wspace': .1})
row_=0
linewidth_=3
for a_station in station_names[:station_count_2_plot]:

    axs[row_].plot(BSI.loc[BSI.Station_Name==a_station, "human_system_start_time"], 
                   BSI.loc[BSI.Station_Name==a_station, "BSI"], 
                   linewidth=linewidth_, ls = '-', label = "BSI", c="k");

    axs[row_].plot(NDSI.loc[NDSI.Station_Name==a_station, "human_system_start_time"], 
                   NDSI.loc[NDSI.Station_Name==a_station, "NDSI"], 
                   linewidth=linewidth_, ls = '-', label = "NDSI", c="r");

    axs[row_].plot(LSWI.loc[LSWI.Station_Name==a_station, "human_system_start_time"], 
                   LSWI.loc[LSWI.Station_Name==a_station, "LSWI"], 
                   linewidth=linewidth_, ls = '-', label = "LSWI", c="g");

    axs[row_].plot(SNOTEL_Snow_depth_2013['Date'], 
                   SNOTEL_Snow_depth_2013[a_station], 
                   linewidth=linewidth_, ls = '-', label = "SNOTEL", c="dodgerblue");

    ax2.tick_params(axis = 'y', which = 'major')
    ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) #
    plt.ylim([-2, 4])

    axs[row_].legend(loc="upper left");
    axs[row_].grid(True);
    axs[row_].set_title(a_station, fontsize=15, fontweight='bold', loc='left');
    axs[row_].set_ylim([-2, 3])
    row_+=1


file_name = snow_dir + "no_pattern"
# plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
# plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)
plt.show()

# %%

# %% [markdown]
# ## Smoothen and plot

# %%
mean = 0
minimizing_variance = 0.2
convolved_BSI = BSI.copy()

# Convolve all the satellite signals and replace
for a_station in station_names:
    a_signal = np.array(BSI.loc[BSI.Station_Name==a_station, "BSI"])
    t = (np.linspace(-10, 10, len(a_signal)) - mean ) / minimizing_variance
    gaussian = (1/minimizing_variance*np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
    gaussian /= np.trapz(gaussian) # normalize the integral to 1
    convolved_signal = np.convolve(gaussian, a_signal, mode='same')

    convolved_BSI.loc[convolved_BSI.Station_Name==a_station, "BSI"]=convolved_signal

# %%
mean = 0
minimizing_variance = 0.2
convolved_LSWI = LSWI.copy()

# Convolve all the satellite signals and replace
for a_station in station_names:
    a_signal = np.array(LSWI.loc[LSWI.Station_Name==a_station, "LSWI"])
    t = (np.linspace(-10, 10, len(a_signal)) - mean ) / minimizing_variance
    gaussian = (1/minimizing_variance*np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
    gaussian /= np.trapz(gaussian) # normalize the integral to 1
    convolved_signal = np.convolve(gaussian, a_signal, mode='same')

    convolved_LSWI.loc[convolved_LSWI.Station_Name==a_station, "LSWI"]=convolved_signal

# %%
mean = 0
minimizing_variance = 0.2
convolved_NDSI = NDSI.copy()

# Convolve all the satellite signals and replace 
# the satellite signals with smoothed version convolved_PMW
for a_station in station_names:
    a_signal = np.array(NDSI.loc[NDSI.Station_Name==a_station, "NDSI"])
    t = (np.linspace(-10, 10, len(a_signal)) - mean ) / minimizing_variance
    gaussian = (1/minimizing_variance*np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
    gaussian /= np.trapz(gaussian) # normalize the integral to 1
    convolved_signal = np.convolve(gaussian, a_signal, mode='same')

    convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "NDSI"]=convolved_signal

# %%
station_count_2_plot=10
fig, axs = plt.subplots(station_count_2_plot, 1, 
                        figsize=(10, station_count_2_plot*3),
                        gridspec_kw={'hspace': 0.3, 'wspace': .1})
row_=0
linewidth_=3
for a_station in station_names[:station_count_2_plot]:

    axs[row_].plot(convolved_BSI.loc[convolved_BSI.Station_Name==a_station, "human_system_start_time"], 
                   convolved_BSI.loc[convolved_BSI.Station_Name==a_station, "BSI"], 
                   linewidth=linewidth_, ls = '-', label = "convolved BSI", c="k");

    axs[row_].plot(convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "human_system_start_time"], 
                   convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "NDSI"], 
                   linewidth=linewidth_, ls = '-', label = "convolved NDSI", c="r");

    axs[row_].plot(convolved_LSWI.loc[convolved_LSWI.Station_Name==a_station, "human_system_start_time"], 
                   convolved_LSWI.loc[convolved_LSWI.Station_Name==a_station, "LSWI"], 
                   linewidth=linewidth_, ls = '-', label = "convolved LSWI", c="g");

    axs[row_].plot(SNOTEL_Snow_depth_2013['Date'], 
                   SNOTEL_Snow_depth_2013[a_station], 
                   linewidth=linewidth_, ls = '-', label = "SNOTEL", c="dodgerblue");

    ax2.tick_params(axis = 'y', which = 'major')
    ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) #
    plt.ylim([-2, 4])

    axs[row_].legend(loc="upper left");
    axs[row_].grid(True);
    axs[row_].set_title(a_station, fontsize=12, loc='left');# fontweight='bold',
    axs[row_].set_ylim([-2, 3])
    row_+=1

file_name = snow_dir + "no_pattern"
# plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
# plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)
plt.show()

# %% [markdown]
# # Interpolate daily NDSI

# %%
convolved_NDSI.head(2)

# %%
convolved_NDSI_notDaily = convolved_NDSI.copy()

# %%
field_IDs = convolved_NDSI.Station_Name.unique()
number_of_fields = len(field_IDs)

base = pd.to_datetime("2013-01-1")
full_year_date_list = [base + datetime.timedelta(days=x) for x in range(365)]

field_IDs_repeated_4_full_year = np.repeat(field_IDs, 365)
calendar_repeated_4_full_year = np.tile(full_year_date_list, number_of_fields)

convolved_NDSI = pd.DataFrame()
convolved_NDSI["Station_Name"] = field_IDs_repeated_4_full_year
convolved_NDSI["human_system_start_time"] = calendar_repeated_4_full_year

convolved_NDSI = pd.merge(convolved_NDSI, 
                          convolved_NDSI_notDaily[["Station_Name", "human_system_start_time", "NDSI"]],
                          on=['Station_Name', 'human_system_start_time'], how='outer')

convolved_NDSI["DoY"] = convolved_NDSI.human_system_start_time.dt.dayofyear 
convolved_NDSI.head(14)

# %% [markdown]
# ### Start and End of the year
#  For some fields, such as ```100457_WSDA_SF_2020``` the first data we have is for Feb. 2.
#  I replace Jan 1 - Feb. 2 with the same value as Feb. 2. Interpolation is not doable!
#  I can extrapolate I guess! That would not be good tho!
#  So, lets take of beginning and end of the year and fill those in!

# %%
import scipy.interpolate

for an_station in convolved_NDSI.Station_Name.unique():
    curr_DF_subset = convolved_NDSI[convolved_NDSI.Station_Name==an_station]
    
    slice_idx_min = curr_DF_subset.index[0]
    slice_idx_max = curr_DF_subset.index[-1]

    # find index of firs NA and last NA and fill whatever NA
    # that there is before and after them
    
    ####
    #### 
    ####
    first_notNA_idx = curr_DF_subset.NDSI.first_valid_index()
    last_notNA_idx  = curr_DF_subset.NDSI.last_valid_index()
    
    first_notNA_value = curr_DF_subset.loc[first_notNA_idx]["NDSI"]
    last_notNA_value  = curr_DF_subset.loc[last_notNA_idx]["NDSI"]
    
    ### Replace potential NAs at the beginning and end:
    convolved_NDSI.loc[slice_idx_min:first_notNA_idx, "NDSI"] = first_notNA_value
    convolved_NDSI.loc[last_notNA_idx:slice_idx_max, "NDSI"] = last_notNA_value
    
    ####
    ####   interpolate between the gaps now
    ####
    # redo the following line so we have an updated version!
    curr_DF_subset = convolved_NDSI[convolved_NDSI.Station_Name==an_station]
    
    ###
    ###    interpolate 
    ###
    # subset non-NAs for interpolation
    not_NA_idx = curr_DF_subset[curr_DF_subset['NDSI'].notnull()]
    not_NA_idx = not_NA_idx.index
    
    # interpolate equations
    x = curr_DF_subset.loc[not_NA_idx, "DoY"].values
    y = curr_DF_subset.loc[not_NA_idx, "NDSI"].values
    y_interp_model = scipy.interpolate.interp1d(x, y)
    
    # interpolate everything
    y_interps = y_interp_model(curr_DF_subset.DoY.values)
    convolved_NDSI.loc[curr_DF_subset.index, "NDSI"] = y_interps

# %%
print (convolved_NDSI.shape)
convolved_NDSI_notDaily.shape

# %%
# first day that NDSI is negative or second or 3rd or 4th:
NDSI_negativeCount = [1, 2, 3, 4, 5]
col_names = ["NDSI_" + str(a_var) + "_negativeCount" for  a_var in NDSI_negativeCount]

predict_lack_of_snow_NDSI = pd.DataFrame(index=range(0, len(station_names)), 
                                          columns = ["Station_Name"] + col_names)

predict_lack_of_snow_NDSI.Station_Name = station_names

for a_var in NDSI_negativeCount:
    predict_lack_of_snow_NDSI["NDSI_" + str(a_var) + "_negativeCount"] = \
      pd.to_datetime(predict_lack_of_snow_NDSI["NDSI_" + str(a_var) + "_negativeCount"])

print (predict_lack_of_snow_NDSI.shape)
print (predict_lack_of_snow_NDSI.head(1))
print ("===========================================================================================")

threshold = -0.3
for a_station in station_names:
    curr_df = convolved_NDSI[convolved_NDSI.Station_Name==a_station]
    first_4_negatives_idx = np.where(curr_df.NDSI<threshold)[0][0:len(NDSI_negativeCount)]
    LL = list(curr_df.iloc[first_4_negatives_idx].human_system_start_time)
    predict_lack_of_snow_NDSI.loc[predict_lack_of_snow_NDSI.Station_Name==a_station, "NDSI_1_negativeCount":]=LL
    
predict_lack_of_snow_NDSI = pd.merge(predict_lack_of_snow_NDSI, 
                                     DayRule28_no_snow_first_DoY[["Station_Name", "no_snow_first_DoY"]], 
                                     on=['Station_Name'], how='left')


predict_errors = predict_lack_of_snow_NDSI.copy()

for a_var in NDSI_negativeCount:
    predict_errors["NDSI_" + str(a_var) + "_negativeCount"] = \
      predict_errors.no_snow_first_DoY - predict_errors["NDSI_" + str(a_var) + "_negativeCount"]

# predict_errors.loc[:, 'NDSI_1_negativeCount':'NDSI_5_negativeCount']=\
#                    np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount':'NDSI_5_negativeCount'])

predict_errors.head(2)

# %%
print ("---------------       min          ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).min())
print ("---------------       mean         ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).mean())
print ("---------------       Max.         ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).max())

# %%

# %%
for a_var in NDSI_negativeCount:
    predict_errors["NDSI_" + str(a_var) + "_negativeCount"] = \
      predict_errors["NDSI_" + str(a_var) + "_negativeCount"].dt.days

predict_errors = pd.merge(predict_errors, PMW[["Station_Name", "longitude", "latitude"]], 
                            on=['Station_Name'], how='left')

output_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/NDSI_plots/"
os.makedirs(output_dir, exist_ok=True)

out_name = output_dir + "predict_errors_NDSI.csv"
predict_errors.to_csv(out_name, index = False)

# %%
a_station = "Mud Ridge"

fig, ax2 = plt.subplots(1, 1, figsize=(15, 5), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.scatter(NDSI.loc[NDSI.Station_Name==a_station, "human_system_start_time"], 
            NDSI.loc[NDSI.Station_Name==a_station, "NDSI"], 
            marker='o', s=20, c='r', label="NDSI")

ax2.plot(convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "human_system_start_time"], 
         convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "NDSI"], 
         linewidth=linewidth_, ls = '-', label = "convolved NDSI", c="r");

ax2.plot(SNOTEL_Snow_depth_2013['Date'], 
         SNOTEL_Snow_depth_2013[a_station], 
         linewidth = 2, ls = '-', label = "SNOTEL", c="dodgerblue");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major')
plt.ylim([-2, 4])

ax2.legend(loc="upper left");
ax2.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

# plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
# plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# %%
n_bins = 300
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(np.array(predict_errors.NDSI_1_negativeCount), n_bins)

ax.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)

plt.xlabel("bins")
plt.ylabel("error (in # of days)")
plt.title("First day NDSI is <" + str(threshold) +" is the no-snow day", 
          fontsize=15, fontweight='bold', loc='left');

out_dir = snow_dir + "NDSI_plots/"
os.makedirs(out_dir, exist_ok=True)
file_name = out_dir + "firstDay_NDSI_lessThanThreshold_isNoSnowDay"
plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# Show plot
plt.show();

# %%
n_bins = 300
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(np.array(predict_errors.NDSI_3_negativeCount), n_bins)

ax.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)

plt.xlabel("X-axis")
plt.ylabel("error (in # of days)")
plt.title("First day NDSI is <" + str(threshold) +" is the no-snow day", 
          fontsize=15, fontweight='bold', loc='left');

out_dir = snow_dir + "NDSI_plots/"
os.makedirs(out_dir, exist_ok=True)
file_name = out_dir + "thirdDay_NDSI_lessThanThreshold_isNoSnowDay"
plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# Show plot
plt.show();

# %% [markdown]
# ## Limit to After Feb?

# %%

# %%
# After Feb. Before Nov.
left_end = pd.to_datetime("2013-02-01")
convolved_NDSI_subset = convolved_NDSI[convolved_NDSI.human_system_start_time>left_end].copy()

right_end = pd.to_datetime("2013-11-01")
convolved_NDSI_subset = convolved_NDSI_subset[\
                                    convolved_NDSI_subset.human_system_start_time<right_end].copy()

# first day that NDSI is negative or second or 3rd or 4th:
NDSI_negativeCount = [1, 2, 3, 4, 5]
col_names = ["NDSI_" + str(a_var) + "_negativeCount" for  a_var in NDSI_negativeCount]

predict_lack_of_snow_NDSI = pd.DataFrame(index=range(0, len(station_names)), 
                                         columns = ["Station_Name"] + col_names)

predict_lack_of_snow_NDSI.Station_Name = station_names

# Change data type:
for a_var in NDSI_negativeCount:
    predict_lack_of_snow_NDSI["NDSI_" + str(a_var) + "_negativeCount"] = \
      pd.to_datetime(predict_lack_of_snow_NDSI["NDSI_" + str(a_var) + "_negativeCount"])

print (predict_lack_of_snow_NDSI.shape)
print (predict_lack_of_snow_NDSI.head(2))

for a_station in station_names:
    curr_df = convolved_NDSI_subset[convolved_NDSI_subset.Station_Name==a_station]
    first_negatives_idx = np.where(curr_df.NDSI<threshold)[0][0:len(NDSI_negativeCount)]
    LL = list(curr_df.iloc[first_negatives_idx].human_system_start_time)
    predict_lack_of_snow_NDSI.loc[predict_lack_of_snow_NDSI.Station_Name==a_station, "NDSI_1_negativeCount":]=LL
    
predict_lack_of_snow_NDSI = pd.merge(predict_lack_of_snow_NDSI, 
                                     DayRule28_no_snow_first_DoY[["Station_Name", "no_snow_first_DoY"]], 
                                     on=['Station_Name'], how='left')
predict_lack_of_snow_NDSI.head(3)

# %%
predict_errors = predict_lack_of_snow_NDSI.copy()

for a_var in NDSI_negativeCount:
    predict_errors["NDSI_" + str(a_var) + "_negativeCount"] = \
      predict_errors.no_snow_first_DoY - predict_errors["NDSI_" + str(a_var) + "_negativeCount"]
    
predict_errors.loc[:, 'NDSI_1_negativeCount':'NDSI_5_negativeCount']=\
             predict_errors.loc[:, 'NDSI_1_negativeCount':'NDSI_5_negativeCount']

predict_errors.head(2)

# %%
print ("---------------       min          ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).min())
print ("---------------       mean         ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).mean())
print ("---------------       Max.         ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).max())

# %%
for a_var in NDSI_negativeCount:
    predict_errors["NDSI_" + str(a_var) + "_negativeCount"] = \
      predict_errors["NDSI_" + str(a_var) + "_negativeCount"].dt.days

# %%
n_bins = 300
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(np.array(predict_errors.NDSI_3_negativeCount), n_bins)

ax.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)

plt.xlabel("X-axis")
plt.ylabel("error (in # of days)")
plt.title("First day NDSI is <" + str(threshold) +" is the no-snow day", 
          fontsize=15, fontweight='bold', loc='left');

out_dir = snow_dir + "NDSI_plots/"
os.makedirs(out_dir, exist_ok=True)
file_name = out_dir + "2ndDay_NDSI_lessThanThreshold_isNoSnowDay_After_Feb_Before_Nov"
plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# Show plot
plt.show();

# %%
predict_errors = pd.merge(predict_errors, PMW[["Station_Name", "longitude", "latitude"]], 
                          on=['Station_Name'], how='left')

output_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/NDSI_plots/"
os.makedirs(output_dir, exist_ok=True)

out_name = output_dir + "predict_errors_NDSI_limitAfterFebBeforeNov.csv"
predict_errors.to_csv(out_name, index = False)

# %%
a_station = "Burnt Mountain"

fig, ax2 = plt.subplots(1, 1, figsize=(15, 5), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.scatter(NDSI.loc[NDSI.Station_Name==a_station, "human_system_start_time"], 
            NDSI.loc[NDSI.Station_Name==a_station, "NDSI"], 
            marker='o', s=20, c='r', label="NDSI")

ax2.plot(convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "human_system_start_time"], 
         convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "NDSI"], 
         linewidth=linewidth_, ls = '-', label = "convolved NDSI", c="r");

ax2.plot(SNOTEL_Snow_depth_2013['Date'], 
         SNOTEL_Snow_depth_2013[a_station], 
         linewidth = 2, ls = '-', label = "SNOTEL", c="dodgerblue");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major')
plt.ylim([-2, 15])

ax2.legend(loc="upper left");
ax2.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

# plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
# plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# %% [markdown]
# # March - June?

# %%
threshold = -0.24
# After Feb. Before Nov.
left_end = pd.to_datetime("2013-03-01")
convolved_NDSI_subset = convolved_NDSI[convolved_NDSI.human_system_start_time>left_end].copy()

right_end = pd.to_datetime("2013-08-1")
convolved_NDSI_subset = convolved_NDSI_subset[\
                                    convolved_NDSI_subset.human_system_start_time<right_end].copy()

# first day that NDSI is negative or second or 3rd or 4th:
NDSI_negativeCount = [1, 2, 3, 4, 5]
col_names = ["NDSI_" + str(a_var) + "_negativeCount" for  a_var in NDSI_negativeCount]

predict_lack_of_snow_NDSI = pd.DataFrame(index=range(0, len(station_names)), 
                                         columns = ["Station_Name"] + col_names)

predict_lack_of_snow_NDSI.Station_Name = station_names

# Change data type:
for a_var in NDSI_negativeCount:
    predict_lack_of_snow_NDSI["NDSI_" + str(a_var) + "_negativeCount"] = \
      pd.to_datetime(predict_lack_of_snow_NDSI["NDSI_" + str(a_var) + "_negativeCount"])

print (predict_lack_of_snow_NDSI.shape)
print (predict_lack_of_snow_NDSI.head(2))

for a_station in station_names:
    curr_df = convolved_NDSI_subset[convolved_NDSI_subset.Station_Name==a_station]
    first_negatives_idx = np.where(curr_df.NDSI<threshold)[0][0:len(NDSI_negativeCount)]
    LL = list(curr_df.iloc[first_negatives_idx].human_system_start_time)
    predict_lack_of_snow_NDSI.loc[predict_lack_of_snow_NDSI.Station_Name==a_station, "NDSI_1_negativeCount":]=LL
    
predict_lack_of_snow_NDSI = pd.merge(predict_lack_of_snow_NDSI, 
                                     DayRule28_no_snow_first_DoY[["Station_Name", "no_snow_first_DoY"]], 
                                     on=['Station_Name'], how='left')
predict_lack_of_snow_NDSI.head(3)

# %%
predict_errors = predict_lack_of_snow_NDSI.copy()

for a_var in NDSI_negativeCount:
    predict_errors["NDSI_" + str(a_var) + "_negativeCount"] = \
      predict_errors.no_snow_first_DoY - predict_errors["NDSI_" + str(a_var) + "_negativeCount"]
    
predict_errors.loc[:, 'NDSI_1_negativeCount':'NDSI_5_negativeCount']=\
             predict_errors.loc[:, 'NDSI_1_negativeCount':'NDSI_5_negativeCount']

print (predict_errors.head(2))

print ("---------------       min          ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).min())
print ("---------------       mean         ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).mean())
print ("---------------       Max.         ------------------------------------------")
print (np.abs(predict_errors.loc[:, 'NDSI_1_negativeCount': 'NDSI_5_negativeCount']).max())

# %%

# %%
n_bins = 200
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(np.array(predict_errors.NDSI_3_negativeCount.dt.days), n_bins)

ax.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)

plt.xlabel("X-axis")
plt.ylabel("error (in # of days)")
plt.title("First day NDSI is <" + str(threshold) +" is the no-snow day", 
          fontsize=15, fontweight='bold', loc='left');

out_dir = snow_dir + "NDSI_plots/"
os.makedirs(out_dir, exist_ok=True)
file_name = out_dir + "2ndDay_NDSI_lessThanThreshold_isNoSnowDay_After_" + \
                      str(left_end.date()) + "_Before_" + str(right_end.date())
plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# Show plot
plt.show();

# %%

# %%
for a_var in NDSI_negativeCount:
    predict_errors["NDSI_" + str(a_var) + "_negativeCount"] = \
      predict_errors["NDSI_" + str(a_var) + "_negativeCount"].dt.days

predict_errors = pd.merge(predict_errors, PMW[["Station_Name", "longitude", "latitude"]], 
                          on=['Station_Name'], how='left')

output_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/NDSI_plots/"
os.makedirs(output_dir, exist_ok=True)

# out_name = output_dir + "predict_errors_NDSI_limitAfter_March_Before_June.csv"
# predict_errors.to_csv(out_name, index = False)

# %% [markdown]
# ### Three consecutive negative NDSI means snow free?

# %%
NDSI.head(2)

# %%
window_size = 4
pred_no_snow_NDSI_consecNegs = pd.DataFrame(index=range(0, len(station_names)), 
                                                  columns = ["Station_Name", "no_snow_date_pred"])
pred_no_snow_NDSI_consecNegs.Station_Name = station_names
pred_no_snow_NDSI_consecNegs["no_snow_date_pred"] = pd.to_datetime(pred_no_snow_NDSI_consecNegs["no_snow_date_pred"])
pred_no_snow_NDSI_consecNegs.head(2)

for a_station in station_names:
    curr_dt = NDSI[NDSI.Station_Name==a_station]
    for curr_row in np.arange(0, curr_dt.shape[0]-window_size):
        curr_window = curr_dt.iloc[curr_row:curr_row+window_size]["NDSI"]
        if np.all(curr_window<0):
            pred_no_snow_NDSI_consecNegs.loc[pred_no_snow_NDSI_consecNegs.Station_Name==a_station, \
                                             "no_snow_date_pred"]=\
                                          curr_dt.loc[curr_window.index[0], "human_system_start_time"]
            break

pred_no_snow_NDSI_consecNegs = pd.merge(pred_no_snow_NDSI_consecNegs, 
                                        DayRule28_no_snow_first_DoY[["Station_Name", "no_snow_first_DoY"]], 
                                        on=['Station_Name'], how='left')

pred_no_snow_NDSI_consecNegs["error"] = predict_errors.no_snow_first_DoY - \
                                        pred_no_snow_NDSI_consecNegs["no_snow_date_pred"]

print ("-------------------    Min.   -------------------")
print ()
print (str(np.abs(pred_no_snow_NDSI_consecNegs.loc[:, 'error']).min().days) + " days error")
print ()
print ("-------------------    mean   -------------------")
print ()
print (str(np.abs(pred_no_snow_NDSI_consecNegs.loc[:, 'error']).mean().days) + " days error")
print ()
print ("-------------------    Max.   -------------------")
print ()
print (str(np.abs(pred_no_snow_NDSI_consecNegs.loc[:, 'error']).max().days) + " days error")

# %%
np.abs(pred_no_snow_NDSI_consecNegs.loc[:, 'error']).max().days

# %%
pred_no_snow_NDSI_consecNegs[pred_no_snow_NDSI_consecNegs.error==\
                             np.abs(pred_no_snow_NDSI_consecNegs.loc[:, 'error']).max()]

# %%
a_station = "Mount Crag"

fig, ax2 = plt.subplots(1, 1, figsize=(15, 5), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.scatter(NDSI.loc[NDSI.Station_Name==a_station, "human_system_start_time"], 
            NDSI.loc[NDSI.Station_Name==a_station, "NDSI"], 
            marker='o', s=20, c='r', label="NDSI")

ax2.plot(convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "human_system_start_time"], 
         convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "NDSI"], 
         linewidth=linewidth_, ls = '-', label = "convolved NDSI", c="r");

ax2.plot(SNOTEL_Snow_depth_2013['Date'], 
         SNOTEL_Snow_depth_2013[a_station], 
         linewidth = 2, ls = '-', label = "SNOTEL", c="dodgerblue");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major')
plt.ylim([-2, 10])

ax2.legend(loc="upper left");
ax2.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

out_dir = snow_dir + "NDSI_plots/"
os.makedirs(out_dir, exist_ok=True)
file_name = out_dir + "ill-behaved-NDSI"

plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# %%
pred_no_snow_NDSI_consecNegs[pred_no_snow_NDSI_consecNegs.error=="0 days"]

# %%
dt = pred_no_snow_NDSI_consecNegs[pred_no_snow_NDSI_consecNegs.error=="0 days"]
station_count_2_plot=dt.shape[0]
fig, axs = plt.subplots(station_count_2_plot, 1, 
                        figsize=(10, station_count_2_plot*3),
                        gridspec_kw={'hspace': 0.3, 'wspace': .1})
row_=0
linewidth_=3
for a_station in dt.Station_Name:

    axs[row_].scatter(NDSI.loc[NDSI.Station_Name==a_station, "human_system_start_time"], 
                      NDSI.loc[NDSI.Station_Name==a_station, "NDSI"], 
                      marker='o', s=20, c='r', label="NDSI")

    axs[row_].plot(convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "human_system_start_time"], 
                   convolved_NDSI.loc[convolved_NDSI.Station_Name==a_station, "NDSI"], 
                   linewidth=linewidth_, ls = '-', label = "convolved NDSI", c="r");

    axs[row_].plot(SNOTEL_Snow_depth_2013['Date'], 
                   SNOTEL_Snow_depth_2013[a_station], 
                   linewidth = 2, ls = '-', label = "SNOTEL", c="dodgerblue");

    ax2.tick_params(axis = 'y', which = 'major')
    ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) #
    plt.ylim([-2, 4])

    axs[row_].legend(loc="upper left");
    axs[row_].grid(True);
    axs[row_].set_title(a_station, fontsize=12, loc='left');# fontweight='bold',
    axs[row_].set_ylim([-2, 5])
    row_+=1

out_dir = snow_dir + "NDSI_plots/"
os.makedirs(out_dir, exist_ok=True)
file_name = out_dir + "well-behaved-NDSI"
plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)
plt.show()

# %%
# PMW = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013.csv")
PMW.loc[PMW.Station_Name==a_station, "lat_lon"]

# %%
print (PMW.loc[PMW.Station_Name=="Upper Wheeler", "lat_lon"].values[0])
print (PMW.loc[PMW.Station_Name=="Green Lake", "lat_lon"].values[0])
print (PMW.loc[PMW.Station_Name=="Spruce Springs", "lat_lon"].values[0])

# %%
# print (list(sorted(pred_no_snow_NDSI_consecNegs.error)))

# %%
bad = pred_no_snow_NDSI_consecNegs[pred_no_snow_NDSI_consecNegs.error=="61 days"].Station_Name.values[0]
PMW.loc[PMW.Station_Name==bad, "lat_lon"].values[0]

# %%
bad = pred_no_snow_NDSI_consecNegs[pred_no_snow_NDSI_consecNegs.error=="48 days"].Station_Name.values[0]
PMW.loc[PMW.Station_Name==bad, "lat_lon"].values[0]

# %%
worst = pred_no_snow_NDSI_consecNegs[pred_no_snow_NDSI_consecNegs.error==\
                             np.abs(pred_no_snow_NDSI_consecNegs.loc[:, 'error']).max()].Station_Name.values[0]

PMW.loc[PMW.Station_Name==worst, "lat_lon"].values[0]

# %%
predict_lack_of_snow_NDSI

# %%
