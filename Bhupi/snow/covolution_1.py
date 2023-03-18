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
# ### Nov. 7.
#
# First attempt to smoothing snow stuff.
#
# Here I want to try a set of different variances in Gaussian function to smooth the satellite and find the best variance/parameter that minimizes the 2-norm of the difference between smoothed-satellite-signal and SNOTEL data.

# %%

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

# from pprint import pprint
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sb

import sys
import io

from pylab import rcParams
# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

from numpy.fft import rfft, irfft, rfftfreq, ifft
from scipy import fft, fftpack
import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

# %%
snow_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/00/"
PMW_diff_dir = snow_dir + "PMW_difference_data/"
SNOTEL_dir = snow_dir + "SNOTEL_data/"

# %%
DayRule28_no_snow_first_DoY=pd.read_csv(snow_dir + "DayRule28_no_snow_first_DoY.csv")
DayRule28_no_snow_first_DoY.head(2)

# %%
SNOTEL_Snow_depth_2013 = pd.read_csv(SNOTEL_dir + "SNOTEL_Snow_depth_2013.csv")
PMW_goodShape = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013_goodShape.csv")

print (SNOTEL_Snow_depth_2013.shape)
print (PMW_goodShape.shape)

# %%
PMW_goodShape.head(2)

# %% [markdown]
# ### Convolution

# %%
train_stations=list(DayRule28_no_snow_first_DoY[DayRule28_no_snow_first_DoY.set_category=="train"].Station_Name)

# %%
a_station = train_stations[0]
a_signal = np.array(PMW_goodShape[a_station])

mean = 0
variance_ = .2
t = (np.linspace(-10, 10, len(a_signal)) - mean ) / variance_
gaussian = (1/variance_*np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
gaussian /= np.trapz(gaussian) # normalize the integral to 1

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)
ax.plot(gaussian, linewidth = 2, ls = '-', label = 'Gaussian', c="dodgerblue");
ax.legend(loc = "upper right"); # , fontsize=20

# %%

# %%
mean = 0
variance_ = .05
t = (np.linspace(-10, 10, len(a_signal)) - mean ) / variance_
gaussian = (1/variance_*np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
gaussian /= np.trapz(gaussian) # normalize the integral to 1

convolved_signal = np.convolve(gaussian, a_signal, mode='same')

# fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex='col', sharey='row',
#                         gridspec_kw={'hspace': 0.2, 'wspace': .1});
# (ax1, ax2) = axs;

fig, ax2 = plt.subplots(1, 1, figsize=(10, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

#######  subplot 1
# ax1.plot(gaussian, linewidth = 2, ls = '-', label = 'Gaussian, variance_=' + str(variance), c="dodgerblue");
# ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
# ax1.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
# ax1.legend(loc="upper right");
# ax1.grid(True);

#######  subplot 2
ax2.plot(a_signal, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
ax2.plot(convolved_signal, linewidth = 3, ls = '-', label = 'convolved', c="k");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

# %% [markdown]
# ### Do the training with norm-2

# %%
SNOTEL_Snow_depth_2013['Date'] = pd.to_datetime(SNOTEL_Snow_depth_2013['Date'])
PMW_goodShape['Date'] = pd.to_datetime(PMW_goodShape['Date'])
DayRule28_no_snow_first_DoY['no_snow_first_DoY'] = pd.to_datetime(DayRule28_no_snow_first_DoY['no_snow_first_DoY'])

# %%
variances = [0.1, 0.15, 0.2, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
var_col_names = ["variance_" + str(a_var).replace(".", "_") + "_norm2" for a_var in variances]
norm2_cost_df = pd.DataFrame(index=range(0, len(train_stations)), 
                             columns = ["train_Station_Name"] + var_col_names)
norm2_cost_df.train_Station_Name=train_stations
norm2_cost_df.head(3)

for a_variance in variances:
    convolved_PMW = PMW_goodShape.copy()
    
    for a_station in train_stations:
        a_signal = np.array(PMW_goodShape[a_station])
        t = (np.linspace(-10, 10, len(a_signal)) - mean ) / a_variance
        gaussian = (1/a_variance*np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
        gaussian /= np.trapz(gaussian) # normalize the integral to 1
        convolved_signal = np.convolve(gaussian, a_signal, mode='same')
        
        convolved_PMW[a_station]=convolved_signal
        
        start_date=list(DayRule28_no_snow_first_DoY[DayRule28_no_snow_first_DoY.Station_Name==a_station]\
                        ["no_snow_first_DoY"])[0]
        
        end_date = start_date + timedelta(days=27)

        curr_PMW = convolved_PMW.loc[(convolved_PMW.Date>=start_date)&(convolved_PMW.Date<=end_date), \
                                      a_station]
        curr_norm = np.linalg.norm(curr_PMW, ord=2)
        colname = "variance_" + str(a_variance).replace(".", "_") + "_norm2"
        norm2_cost_df.loc[norm2_cost_df.train_Station_Name==a_station, colname]=curr_norm


# %%
df2 = pd.DataFrame(norm2_cost_df.iloc[:, 1:].mean()).T

# %%
df3 = pd.concat([norm2_cost_df, df2], ignore_index = True)
df3.head(3)

# %% [markdown]
# # Apply to test set AND train set
# and check if we get close to reality from SNOTEL. Perhaps the norm-2 is not minimized or maximized when
# there is no snow!

# %%
minimizing_variance = 0.25
convolved_PMW = PMW_goodShape.copy()

# Convolve all the satellite signals and replace 
# the satellite signals with smoothed version convolved_PMW
for a_station in list(DayRule28_no_snow_first_DoY.Station_Name):
    a_signal = np.array(convolved_PMW[a_station])
    t = (np.linspace(-10, 10, len(a_signal)) - mean ) / minimizing_variance
    gaussian = (1/minimizing_variance*np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
    gaussian /= np.trapz(gaussian) # normalize the integral to 1
    convolved_signal = np.convolve(gaussian, a_signal, mode='same')

    convolved_PMW[a_station]=convolved_signal

# %% [markdown]
# #### For each station find the 28-day window in which the norm-2 of convolved signal is minimized!

# %%
convolved_PMW.head(3)

# %%
DayRule28_no_snow_first_DoY_helper=DayRule28_no_snow_first_DoY.copy()
DayRule28_no_snow_first_DoY_helper["minNorm2_first_DoY"] = pd.to_datetime("1984-01-27")
DayRule28_no_snow_first_DoY_helper.head(2)

for a_station in list(DayRule28_no_snow_first_DoY_helper.Station_Name):
    curr_min = float('inf')
    for a_date in list(convolved_PMW.Date[:337]):
        curr_signal = convolved_PMW.loc[(convolved_PMW.Date>=a_date) & \
                                        (convolved_PMW.Date<=a_date+timedelta(days=27)), a_station]
        curr_norm = np.linalg.norm(curr_signal, ord=2)
        if curr_norm<curr_min:
            curr_min=curr_norm
            DayRule28_no_snow_first_DoY_helper.loc[DayRule28_no_snow_first_DoY_helper.Station_Name==a_station, \
                                                   "minNorm2_first_DoY"] = a_date

# %%
DayRule28_no_snow_first_DoY_helper.head(3)

# %%

# %%
row_idx = 0
a_station = DayRule28_no_snow_first_DoY_helper.Station_Name[0]
correct_date = DayRule28_no_snow_first_DoY_helper.no_snow_first_DoY[0]

correct_norm = np.linalg.norm(convolved_PMW.loc[(convolved_PMW.Date>=correct_date) & \
                                        (convolved_PMW.Date<=correct_date+timedelta(days=27)), a_station],
                              ord=2)


print ("norm in correct window: " + str(correct_norm.round(2)))

predict_date = DayRule28_no_snow_first_DoY_helper.minNorm2_first_DoY[0]
correct_norm = np.linalg.norm(convolved_PMW.loc[(convolved_PMW.Date>=predict_date) & \
                                                (convolved_PMW.Date<=predict_date+timedelta(days=27)), a_station],
                             ord=2)
print ("norm in predicted window: " + str(correct_norm.round(2)))

# %%
fig, ax2 = plt.subplots(1, 1, figsize=(15, 5), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(PMW_goodShape["Date"], PMW_goodShape[a_station], 
         linewidth = 2, ls = '-', label = 'PMW signal', c="k");

ax2.plot(PMW_goodShape["Date"], convolved_PMW[a_station], 
         linewidth = 4, ls = '-', label = 'convolved signal', c="dodgerblue");

ax2.plot(SNOTEL_Snow_depth_2013["Date"], SNOTEL_Snow_depth_2013[a_station], 
         linewidth = 3, ls = '-.', label = 'SNOTEL', c="g");

truth_date = pd.to_datetime("2013-03-08")
# plt.axvline(x = truth_date, c="g", linewidth=2, label = 'truth_date')
# plt.axvline(x = truth_date + timedelta(days=27), c="g", linewidth=2)
ax2.axvspan(truth_date, 
            truth_date + timedelta(days=27), 
            alpha=0.3, facecolor="g", label='Truth Date')

predicted_date=pd.to_datetime("2013-08-26")
# plt.axvline(x = predicted_date, c="r", linewidth=2, label = 'predicted_date')
# plt.axvline(x = predicted_date + timedelta(days=27), c="r", linewidth=2)
ax2.axvspan(predicted_date, predicted_date + timedelta(days=27), 
            alpha=0.3, label = 'predicted date', facecolor='dodgerblue')

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) #
# ax2.xaxis.set_major_locator(mdates.MonthLocator())
# ax2.xaxis.set_major_formatter(DateFormatter('%b'))
plt.ylim([-2, 15])

ax2.legend(loc="upper right");
ax2.grid(True);

file_name = snow_dir + "shooting_blind_conv"
# plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
# plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)

# %%

# %%
SNOTEL_Snow_depth_2013.head(3)

# %%

# %%
# Add alphabetic form 
# of no_snow_first_DoY so we can use them for slicing as 
# column names of PMW_difference_data_2013 are months abbreviations etc.

# days_ = DayRule28_no_snow_first_DoY.no_snow_first_DoY.dt.day.astype(str)
# for an_ind in days_.index:
#     if len(days_[an_ind])==1:
#         days_[an_ind] = "0"+days_[an_ind]

# alph_no_snow_first_DoY = DayRule28_no_snow_first_DoY["no_snow_first_DoY"].dt.month_name().str.slice(stop=3) + \
#                          "_" + \
#                          days_ + "_" + \
#                          DayRule28_no_snow_first_DoY["no_snow_first_DoY"].dt.year.astype(str) 

# DayRule28_no_snow_first_DoY["alph_no_snow_first_DoY"] = alph_no_snow_first_DoY
# DayRule28_no_snow_first_DoY.head(2)

# %%

# %%
print ("L2 norm is " + str(np.linalg.norm(curr_PMW, ord=2).round()))
print ("L1 norm is " + str(np.linalg.norm(curr_PMW, ord=1).round()))
print ("L-inf norm is " + str(np.linalg.norm(curr_PMW, ord=np.inf).round()))

# %%
np.linalg.norm(list(curr_PMW), ord=1).round()

# %% [markdown]
# ### Plot 10 different locations
# and compare their PMW vs SNOTEL

# %%
# x=[[1,2,3,4],[1,4,3,4],[1,2,3,4],[9,8,7,4]]
# y=[[3,2,3,4],[3,6,3,4],[6,7,8,9],[3,2,2,4]]

# plots = zip(x,y)
# def loop_plot(plots):
#     figs={}
#     axs={}
#     for idx,plot in enumerate(plots):
#         figs[idx]=plt.figure()
#         axs[idx]=figs[idx].add_subplot(111)
#         axs[idx].plot(plot[0],plot[1])
#     return figs, axs  
        
# figs, axs = loop_plot(plots)

# %%
station_names = list(SNOTEL_Snow_depth_2013.columns[2:])
print (station_names[:4])

import random
random.seed(0)
random.shuffle(station_names)

# %%
DayRule28_no_snow_first_DoY.head(2)

# %%
station_count_2_plot=10
fig, axs = plt.subplots(station_count_2_plot, 1, 
                        figsize=(10, station_count_2_plot*3),
                        gridspec_kw={'hspace': 0.2, 'wspace': .1})
row_=0
for a_station in station_names[:station_count_2_plot]:
    label_ = 'PMW (' + a_station + ')'
    axs[row_].plot(PMW_goodShape["Date"], PMW_goodShape[a_station], 
                  linewidth = 2, ls = '-', label = label_, c="k");

    truth_date = list(DayRule28_no_snow_first_DoY.loc[DayRule28_no_snow_first_DoY.Station_Name==a_station, \
                                                      "no_snow_first_DoY"])[0]
    axs[row_].axvspan(truth_date, 
                      truth_date + timedelta(days=27), 
                      alpha=0.3, facecolor="g", label='Truth Date')
#     axs[row_].plot(SNOTEL_Snow_depth_2013["Date"], 
#                SNOTEL_Snow_depth_2013[a_station], 
#                linewidth = 3, ls = '-.', label = 'SNOTEL', c="g");
#     axs[row_].set_ylim([PMW_goodShape[a_station].min()-0.5, PMW_goodShape[a_station].max()+0.5])

    axs[row_].legend(loc="upper right");
    axs[row_].grid(True);
    row_+=1


file_name = snow_dir + "no_pattern"
# plt.savefig(fname = file_name + ".pdf", dpi=400, bbox_inches='tight', transparent=False)
# plt.savefig(fname = file_name+ ".png", dpi=400, bbox_inches='tight', transparent=False)
plt.show()

# %%

# %%

# %% [markdown]
# # Convolve and Export the result

# %%
convolved_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/02_convolved/"
os.makedirs(convolved_dir, exist_ok=True)

variances = [0.05, 0.1, 0.2]
for a_variance in variances:
    convolved_PMW = PMW_goodShape.copy()
    
    for a_station in train_stations:
        a_signal = np.array(PMW_goodShape[a_station])
        t = (np.linspace(-10, 10, len(a_signal)) - mean ) / a_variance
        gaussian = (1/a_variance*np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
        gaussian /= np.trapz(gaussian) # normalize the integral to 1
        convolved_signal = np.convolve(gaussian, a_signal, mode='same')
        
        convolved_PMW[a_station]=convolved_signal
    
    out_name = convolved_dir + "PMW_convolved_" + str("variance_" + str(a_variance).replace(".", "_")) + ".csv"
    convolved_PMW.to_csv(out_name, index = False)

# %%
out_name

# %%
str("variance_" + str(0.05).replace(".", "_"))

# %%
