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
import datetime
import sys
import io

from pylab import rcParams
# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

from numpy.fft import rfft, irfft, rfftfreq, ifft
from scipy import fft
import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter


# %%
class one_curve_class:
    def __init__(self, x1, y1, ccolor, leg1, x_label, y_label, x_limits):
        self.x1 = x1
        self.y1 = y1
        self.leg1 = leg1
        self.ccolor = ccolor

        self.x_label = x_label
        self.y_label = y_label
        self.x_limits = x_limits

def plot_1curve_in_1subplot(one_curve_obj):
    # plt.subplots_adjust(left=0, bottom=0, right=1.1, top=0.9, wspace=0, hspace=0)
    # plt.subplots_adjust(wspace=.1, hspace = 0.5)
    plt.subplots_adjust(right = 1.5, top = 1.1)

    ##########################################################################################
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 3)
    ax1.grid(True);

    ax1.plot(one_curve_obj.x1, 
             one_curve_obj.y1, 
             '-', 
             linewidth = 3,
             label = one_curve_obj.leg1, 
             c = one_curve_obj.ccolor)

    ax1.set_xlabel(one_curve_obj.y_label) 
    ax1.set_ylabel(one_curve_obj.y_label) # , labelpad=20); # fontsize = label_FontSize,

    ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    ax1.tick_params(axis='x', which='major') #,
    ax1.legend(loc="upper right"); # , fontsize=12
    
    plt.xlim(one_curve_obj.x_limits)
    


# %%
snow_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/00/"
PMW_diff_dir = snow_dir + "PMW_difference_data/"
SNOTEL_dir = snow_dir + "SNOTEL_data/"

# %%
SNOTEL_Snow_depth_2013 = pd.read_csv(SNOTEL_dir + "SNOTEL_Snow_depth_2013.csv")
PMW_goodShape = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013_goodShape.csv")
PMW_goodShape["Date"] = pd.to_datetime(PMW_goodShape["Date"])
print (SNOTEL_Snow_depth_2013.shape)
print (PMW_goodShape.shape)

# %%
DayRule28_no_snow_first_DoY=pd.read_csv(snow_dir + "DayRule28_no_snow_first_DoY.csv")
DayRule28_no_snow_first_DoY["no_snow_first_DoY"] = pd.to_datetime(DayRule28_no_snow_first_DoY["no_snow_first_DoY"])
DayRule28_no_snow_first_DoY.head(2)

# %%
station_names = DayRule28_no_snow_first_DoY.Station_Name.values

# %%
Station_Name = station_names[0]
Station_Name

a_signal = PMW_goodShape[Station_Name].values
noisy_FFT_y = fft.rfft(a_signal)
print (max(np.abs(noisy_FFT_y)))

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(a_signal, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");

# %%

# %% [markdown]
# ### Remove high freq.

# %%
noisy_FFT_yRemove = noisy_FFT_y.copy()
threshold = 153
noisy_FFT_yRemove[np.abs(noisy_FFT_yRemove) < threshold] = 0

# fig, ax = plt.subplots()
# fig.set_size_inches(10, 3)
# ax.plot(# np.arange(len(noisy_FFT_yRemove)), 
#         np.abs(noisy_FFT_yRemove), 
#         '-r', linewidth=3)

# ax.tick_params(axis='y', which='major', labelsize = 15)
# ax.tick_params(axis='x', which='major', labelsize = 15)
# plt.xlim([-5, 190])
# ax.grid(True);

# %% [markdown]
# ## Apply inverse FFT

# %%
noisy_FFT_yRemove = noisy_FFT_y.copy()
threshold = 152.68
noisy_FFT_yRemove[np.abs(noisy_FFT_yRemove) < threshold] = 0

recovered_sig = irfft(noisy_FFT_yRemove)

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(a_signal, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
ax.plot(recovered_sig, 'k', ls='-', linewidth = 3, label = 'recovered signal');
ax.legend(loc = "upper right"); # , fontsize=20

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter('%b'))

file_name = snow_dir + "threshold_152.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)

# %%

noisy_FFT_yRemove = noisy_FFT_y.copy()
threshold = 152.69
noisy_FFT_yRemove[np.abs(noisy_FFT_yRemove) < threshold] = 0

recovered_sig = irfft(noisy_FFT_yRemove)

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(a_signal, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
ax.plot(recovered_sig, 'k', ls='-', linewidth = 3, label = 'recovered signal');

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter('%b'))
ax.legend(loc = "upper right"); # , fontsize=20

file_name = snow_dir + "threshold_153.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)

# %%
given_col = SNOTEL_Snow_depth_2013.columns[5]

SNOTEL_signal = SNOTEL_Snow_depth_2013.loc[:, given_col]
PMW_signal=PMW_goodShape[given_col].values

fig, axs = plt.subplots(2, 1, figsize=(10, 6),
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});
(ax1, ax2) = axs;

#######  subplot 1
# ax1 = plt.subplot(211)
ax1.plot(SNOTEL_signal,linewidth=3, ls='-', label = "SNOTEL "+ given_col , c="g");
ax1.plot(PMW_signal,   linewidth=3, ls='-', label = "PMW " + given_col, c="dodgerblue");

ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(DateFormatter('%b'))

ax1.legend(loc="upper right");
ax1.grid(True);
ax1.set_xlim([0, 230])
ax1.set_ylim([-1, 300])
#######  subplot 2
# ax2 = plt.subplot(212)
ax2.plot(SNOTEL_signal,linewidth=3, ls='-', label = "SNOTEL "+ given_col , c="g");
ax2.plot(PMW_signal,   linewidth=3, ls='-', label = "PMW " + given_col, c="dodgerblue");

ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="best");
ax2.grid(True);

ax2.set_xlim([150, 230]);
ax2.set_ylim([-1, 20]);

file_name = snow_dir + "eg2.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)

# %% [markdown]
# ### Are frequencies different in different months?

# %%

# %%
a_station = SNOTEL_Snow_depth_2013.columns[5]

PMW_signal=PMW_goodShape[a_station].values
SNOTEL_signal = SNOTEL_Snow_depth_2013[a_station].values

fig, axs = plt.subplots(1, 1, figsize=(10, 6),
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

#######  subplot 1
# ax1 = plt.subplot(211)
axs.plot(SNOTEL_signal,linewidth=3, ls='-', label = "SNOTEL "+ given_col , c="g");
axs.plot(PMW_signal,   linewidth=3, ls='-', label = "PMW " + given_col, c="dodgerblue");

axs.xaxis.set_major_locator(mdates.MonthLocator())
axs.xaxis.set_major_formatter(DateFormatter('%b'))

axs.legend(loc="best");
axs.grid(True);

# %%
PMW_FFT = fft.rfft(PMW_signal)

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(np.abs(PMW_FFT), '-', label = 'PMW rFFT', linewidth=3, c="dodgerblue")
# ax.plot(np.abs(PMW_signal), '-', label = 'PMW_signal', linewidth=3, c="red")

ax.set_xlabel('frequency') # , labelpad = 20); # fontsize = label_FontSize,
ax.set_ylabel('amplitude') # , fontsize = 20) # , labelpad=20); # fontsize = label_FontSize,
ax.tick_params(axis='y', which='major') # , labelsize = 15) #) #
ax.tick_params(axis='x', which='major') # , labelsize = 15) #) #
# plt.xlim([-5, 1000])
plt.ylim([-2, 150])

ax.legend(loc = "upper right"); # , fontsize=20
ax.grid(True);

# %%
# x = np.array([1, 2, 1, 0, 1, 2, 1, 0])
x = PMW_signal
  
# compute DFT with optimized FFT
w = np.fft.fft(x)
  
# compute frequency associated
# with coefficients
freqs = np.fft.fftfreq(len(x))
  
# extract frequencies associated with FFT values
for coef, freq in zip(w, freqs):
    if coef:
        print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef,
                                                    f=freq))

# %%
ii = 1
print ("w[0] is ", w[ii])
print ("real part is ", np.real(w[ii]))
print ("imaginary part is ", np.imag(w[ii]))

# %%
PMW_goodShape.head(2)

# %%
a_station = SNOTEL_Snow_depth_2013.columns[5]

PMW_signal=PMW_goodShape[a_station].values
SNOTEL_signal = SNOTEL_Snow_depth_2013[a_station].values

fig, axs = plt.subplots(1, 1, figsize=(10, 6),
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

#######  subplot 1
# ax1 = plt.subplot(211)
axs.plot(SNOTEL_signal,linewidth=3, ls='-', label = "SNOTEL "+ given_col , c="g");
axs.plot(PMW_signal,   linewidth=3, ls='-', label = "PMW " + given_col, c="dodgerblue");

# axs.set_xlim([150, 230]);
axs.set_ylim([-1, 20]);

axs.xaxis.set_major_locator(mdates.MonthLocator())
axs.xaxis.set_major_formatter(DateFormatter('%b'))

axs.legend(loc="best");
axs.grid(True);

# %%
a_station = station_names[3]

PMW_goodShape["Date"] = pd.to_datetime(PMW_goodShape["Date"])
a_station_PMW_dt = PMW_goodShape[["Date", a_station]]
a_station_PMW_dt.head(2)

no_snow_begin = pd.to_datetime(DayRule28_no_snow_first_DoY[DayRule28_no_snow_first_DoY.Station_Name==\
                                                          a_station].no_snow_first_DoY.values[0])
left_ = no_snow_begin-timedelta(days=27)
right_= no_snow_begin+timedelta(days=27)

print ("left_         ", left_)
print ("no_snow_begin ", no_snow_begin)
print ("right_        ", right_)
print ("------------------------------------------------------------------")
snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=left_) & (a_station_PMW_dt.Date<=no_snow_begin)][a_station]
no_snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=no_snow_begin) & (a_station_PMW_dt.Date<=right_)][a_station]

snow_window_FFT = fft.rfft(snow_window.values)
no_snow_window_FFT = fft.rfft(no_snow_window.values)

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(np.abs(snow_window_FFT), '-', label = 'snow rFFT', linewidth=3, c="dodgerblue")
ax.plot(np.abs(no_snow_window_FFT), '-', label = 'no_snow rFFT', linewidth=3, c="red")
# ax.plot(np.abs(PMW_signal), '-', label = 'PMW_signal', linewidth=3, c="red")

ax.set_xlabel('frequency') # , labelpad = 20); # fontsize = label_FontSize,
ax.set_ylabel('amplitude') # , fontsize = 20) # , labelpad=20); # fontsize = label_FontSize,
ax.tick_params(axis='y', which='major') # , labelsize = 15) #) #
ax.tick_params(axis='x', which='major') # , labelsize = 15) #) #
# plt.xlim([-5, 1000])
plt.ylim([-2, 100])

ax.legend(loc = "upper right"); # , fontsize=20
ax.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

# %%

# %%
a_station = station_names[1]

a_station_PMW_dt = PMW_goodShape[["Date", a_station]]
a_station_PMW_dt.head(2)

no_snow_begin = pd.to_datetime(DayRule28_no_snow_first_DoY[DayRule28_no_snow_first_DoY.Station_Name==\
                                                          a_station].no_snow_first_DoY.values[0])
left_ = no_snow_begin-timedelta(days=27)
right_= no_snow_begin+timedelta(days=27)

print ("left_         ", left_)
print ("no_snow_begin ", no_snow_begin)
print ("right_        ", right_)
print ("------------------------------------------------------------------")
snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=left_) & (a_station_PMW_dt.Date<=no_snow_begin)][a_station]
no_snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=no_snow_begin) & (a_station_PMW_dt.Date<=right_)][a_station]

snow_window_FFT = fft.rfft(snow_window.values)
no_snow_window_FFT = fft.rfft(no_snow_window.values)

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(np.abs(snow_window_FFT), '-', label = 'snow rFFT', linewidth=3, c="dodgerblue")
ax.plot(np.abs(no_snow_window_FFT), '-', label = 'no_snow rFFT', linewidth=3, c="red")
# ax.plot(np.abs(PMW_signal), '-', label = 'PMW_signal', linewidth=3, c="red")

ax.set_xlabel('frequency') # , labelpad = 20); # fontsize = label_FontSize,
ax.set_ylabel('amplitude') # , fontsize = 20) # , labelpad=20); # fontsize = label_FontSize,
ax.tick_params(axis='y', which='major') # , labelsize = 15) #) #
ax.tick_params(axis='x', which='major') # , labelsize = 15) #) #
# plt.xlim([-5, 1000])
plt.ylim([-2, 80])

ax.legend(loc = "upper right"); # , fontsize=20
ax.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

# %%
a_station = station_names[11] # 15 is bad too

a_station_PMW_dt = PMW_goodShape[["Date", a_station]]
a_station_PMW_dt.head(2)

no_snow_begin = pd.to_datetime(DayRule28_no_snow_first_DoY[DayRule28_no_snow_first_DoY.Station_Name==\
                                                          a_station].no_snow_first_DoY.values[0])
left_ = no_snow_begin-timedelta(days=27)
right_= no_snow_begin+timedelta(days=27)

snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=left_) & (a_station_PMW_dt.Date<=no_snow_begin)][a_station]
no_snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=no_snow_begin) & (a_station_PMW_dt.Date<=right_)][a_station]

snow_window_FFT = fft.rfft(snow_window.values)
no_snow_window_FFT = fft.rfft(no_snow_window.values)

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(np.abs(snow_window_FFT), '-', label = 'snow rFFT', linewidth=3, c="dodgerblue")
ax.plot(np.abs(no_snow_window_FFT), '-', label = 'no_snow rFFT', linewidth=3, c="red")
# ax.plot(np.abs(PMW_signal), '-', label = 'PMW_signal', linewidth=3, c="red")

ax.set_xlabel('frequency') # , labelpad = 20); # fontsize = label_FontSize,
ax.set_ylabel('amplitude') # , fontsize = 20) # , labelpad=20); # fontsize = label_FontSize,
ax.tick_params(axis='y', which='major') # , labelsize = 15) #) #
ax.tick_params(axis='x', which='major') # , labelsize = 15) #) #
# plt.xlim([-5, 1000])
plt.ylim([-2, 100])

ax.legend(loc = "upper right"); # , fontsize=20
ax.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

# %%
a_station = station_names[16]

a_station_PMW_dt = PMW_goodShape[["Date", a_station]]
a_station_PMW_dt.head(2)

no_snow_begin = pd.to_datetime(DayRule28_no_snow_first_DoY[DayRule28_no_snow_first_DoY.Station_Name==\
                                                          a_station].no_snow_first_DoY.values[0])
left_ = no_snow_begin-timedelta(days=27)
right_= no_snow_begin+timedelta(days=27)

snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=left_) & (a_station_PMW_dt.Date<=no_snow_begin)][a_station]
no_snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=no_snow_begin) & (a_station_PMW_dt.Date<=right_)][a_station]

snow_window_FFT = fft.rfft(snow_window.values)
no_snow_window_FFT = fft.rfft(no_snow_window.values)

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(np.abs(snow_window_FFT), '-', label = 'snow rFFT', linewidth=3, c="dodgerblue")
ax.plot(np.abs(no_snow_window_FFT), '-', label = 'no_snow rFFT', linewidth=3, c="red")
# ax.plot(np.abs(PMW_signal), '-', label = 'PMW_signal', linewidth=3, c="red")

ax.set_xlabel('frequency') # , labelpad = 20); # fontsize = label_FontSize,
ax.set_ylabel('amplitude') # , fontsize = 20) # , labelpad=20); # fontsize = label_FontSize,
ax.tick_params(axis='y', which='major') # , labelsize = 15) #) #
ax.tick_params(axis='x', which='major') # , labelsize = 15) #) #
# plt.xlim([-5, 1000])
plt.ylim([-2, 100])

ax.legend(loc = "upper right"); # , fontsize=20
ax.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

# %% [markdown]
# # other time of the year with all snow:

# %%
a_station = station_names[15]

a_station_PMW_dt = PMW_goodShape[["Date", a_station]]

no_snow_begin = pd.to_datetime(DayRule28_no_snow_first_DoY[DayRule28_no_snow_first_DoY.Station_Name==\
                                                          a_station].no_snow_first_DoY.values[0])

no_snow_begin = pd.to_datetime("2013-02-02")
left_ = no_snow_begin-timedelta(days=27)
right_= no_snow_begin+timedelta(days=27)

snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=left_) & (a_station_PMW_dt.Date<=no_snow_begin)][a_station]
no_snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=no_snow_begin) & (a_station_PMW_dt.Date<=right_)][a_station]
snow_window_FFT = fft.rfft(snow_window.values)
no_snow_window_FFT = fft.rfft(no_snow_window.values)

fig, ax = plt.subplots()
fig.set_size_inches(10, 3)

ax.plot(np.abs(snow_window_FFT), '-', label = 'snow rFFT', linewidth=3, c="dodgerblue")
ax.plot(np.abs(no_snow_window_FFT), '-', label = 'no_snow rFFT', linewidth=3, c="red")
# ax.plot(np.abs(PMW_signal), '-', label = 'PMW_signal', linewidth=3, c="red")

ax.set_xlabel('frequency') # , labelpad = 20); # fontsize = label_FontSize,
ax.set_ylabel('amplitude') # , fontsize = 20) # , labelpad=20); # fontsize = label_FontSize,
ax.tick_params(axis='y', which='major') # , labelsize = 15) #) #
ax.tick_params(axis='x', which='major') # , labelsize = 15) #) #
# plt.xlim([-5, 1000])
plt.ylim([-2, 100])

ax.legend(loc = "upper right"); # , fontsize=20
ax.grid(True);
plt.title(a_station, fontsize=15, fontweight='bold', loc='left');

# %% [markdown]
# # Smooth (Convolve) then FFT

# %%
convolved_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/02_convolved/"
convolved_PMW_0_2=pd.read_csv(convolved_dir+"PMW_convolved_variance_0_05.csv")
convolved_PMW_0_2["Date"] = pd.to_datetime(convolved_PMW_0_2["Date"])

# %%
a_station = station_names[0]

no_snow_begin = pd.to_datetime(DayRule28_no_snow_first_DoY[DayRule28_no_snow_first_DoY.Station_Name==\
                                                          a_station].no_snow_first_DoY.values[0])

left_ = no_snow_begin-timedelta(days=27)
right_= no_snow_begin+timedelta(days=27)

fig, axs = plt.subplots(2, 1, figsize=(10, 6),
                        gridspec_kw={'hspace': 0.3, 'wspace': .1});
(ax1, ax2) = axs;

a_station_PMW_dt = PMW_goodShape[["Date", a_station]]
snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=left_) & (a_station_PMW_dt.Date<=no_snow_begin)][a_station]
no_snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=no_snow_begin) & (a_station_PMW_dt.Date<=right_)][a_station]
snow_window_FFT = fft.rfft(snow_window.values)
no_snow_window_FFT = fft.rfft(no_snow_window.values)

ax1.plot(np.abs(snow_window_FFT), '-', label = 'snow rFFT', linewidth=3, c="dodgerblue")
ax1.plot(np.abs(no_snow_window_FFT), '-', label = 'no_snow rFFT', linewidth=3, c="red")
ax1.set_xlabel('frequency') # , labelpad = 20); # fontsize = label_FontSize,
ax1.set_ylabel('amplitude') # , fontsize = 20) # , labelpad=20); # fontsize = label_FontSize,
ax1.tick_params(axis='y', which='major') # , labelsize = 15) #) #
ax1.tick_params(axis='x', which='major') # , labelsize = 15) #) #
# plt.xlim([-5, 1000])
ax1.set_ylim([-2, 100])

ax1.legend(loc = "upper right"); # , fontsize=20
ax1.grid(True);
ax1.set_title(a_station, fontsize=15, loc='left'); # fontweight='bold',

###############################################################################################

a_station_PMW_dt = convolved_PMW_0_2[["Date", a_station]]
snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=left_) & (a_station_PMW_dt.Date<=no_snow_begin)][a_station]
no_snow_window = a_station_PMW_dt[(a_station_PMW_dt.Date>=no_snow_begin) & (a_station_PMW_dt.Date<=right_)][a_station]
snow_window_FFT = fft.rfft(snow_window.values)
no_snow_window_FFT = fft.rfft(no_snow_window.values)

ax2.plot(np.abs(snow_window_FFT), '-', label = 'snow rFFT', linewidth=3, c="dodgerblue")
ax2.plot(np.abs(no_snow_window_FFT), '-', label = 'no_snow rFFT', linewidth=3, c="red")
ax2.set_xlabel('frequency') # , labelpad = 20); # fontsize = label_FontSize,
ax2.set_ylabel('amplitude') # , fontsize = 20) # , labelpad=20); # fontsize = label_FontSize,
ax2.tick_params(axis='y', which='major') # , labelsize = 15) #) #
ax2.tick_params(axis='x', which='major') # , labelsize = 15) #) #
# plt.xlim([-5, 1000])
ax2.set_ylim([-2, 100])

ax2.legend(loc = "upper right"); # , fontsize=20
ax2.grid(True);
ax2.set_title(a_station + ' (convolved)', fontsize=15, loc='left'); # fontweight='bold',

# %% [markdown]
# ### Compare FFt in 3-, 4-, 5- day window and see what happens
#
#  - First form FFT table

# %%
FFT_table = pd.DataFrame(index=range(0, 183), 
                         columns = station_names)

for a_station in station_names:
    FFT_table[a_station]= fft.rfft(PMW_goodShape[a_station].values)

# %%
window_sizes = [3, 4, 5]
win_size_col_names = ["win_size_" + str(win_size) for win_size in window_sizes]
windows_df = pd.DataFrame(index=range(0, len(station_names)), 
                         columns = ["Station_Name"] + win_size_col_names)
windows_df.Station_Name=station_names
windows_df.head(3)

gap_size = 0
for a_station in station_names:
    for a_window in window_sizes:
        for a_row in np.arange(0,183-a_window*2-gap_size+1):
            early_window = FFT_table.loc[a_row:a_row+a_window-1, a_station]
            late_window = FFT_table.loc[a_row+a_window+gap_size : a_row+a_window+gap_size+a_window-1, a_station]
            diff = np.abs(early_window).values - np.abs(late_window).values
            if np.all(diff>=0):
                windows_df.loc[windows_df.Station_Name==a_station, "win_size_" + str(a_window)]=a_row

windows_df = pd.merge(windows_df, 
                      DayRule28_no_snow_first_DoY[["Station_Name", "no_snow_first_DoY"]], 
                      on=['Station_Name'], how='left')
windows_df['DoY'] = 666

for a_idx in windows_df.index:
    windows_df.loc[a_idx, "DoY"] = windows_df.loc[a_idx, "no_snow_first_DoY"].day_of_year

windows_df    

# %%
window_sizes = [3, 4, 5]
win_size_col_names = ["win_size_" + str(win_size) for win_size in window_sizes]
windows_df = pd.DataFrame(index=range(0, len(station_names)), 
                         columns = ["Station_Name"] + win_size_col_names)
windows_df.Station_Name=station_names
windows_df.head(3)

gap_size = 3
for a_station in station_names:
    for a_window in window_sizes:
        for a_row in np.arange(0,183-a_window*2-gap_size+1):
            early_window = FFT_table.loc[a_row:a_row+a_window-1, a_station]
            late_window = FFT_table.loc[a_row+a_window+gap_size : a_row+a_window+gap_size+a_window-1, a_station]
            diff = np.abs(early_window).values - np.abs(late_window).values
            if np.all(diff>=0):
                windows_df.loc[windows_df.Station_Name==a_station, "win_size_" + str(a_window)]=a_row

windows_df = pd.merge(windows_df, 
                      DayRule28_no_snow_first_DoY[["Station_Name", "no_snow_first_DoY"]], 
                      on=['Station_Name'], how='left')
windows_df['DoY'] = 666

for a_idx in windows_df.index:
    windows_df.loc[a_idx, "DoY"] = windows_df.loc[a_idx, "no_snow_first_DoY"].day_of_year

windows_df    

# %%
window_sizes = [3, 4, 5]
win_size_col_names = ["win_size_" + str(win_size) for win_size in window_sizes]
windows_df = pd.DataFrame(index=range(0, len(station_names)), 
                         columns = ["Station_Name"] + win_size_col_names)
windows_df.Station_Name=station_names
windows_df.head(3)

gap_size=2
for a_station in station_names:
    for a_window in window_sizes:
        for a_row in np.arange(0,183-a_window*2-gap_size+1):
            early_window = FFT_table.loc[a_row:a_row+a_window-1, a_station]
            late_window = FFT_table.loc[a_row+a_window+gap_size : a_row+a_window+gap_size+a_window-1, a_station]
            diff = np.abs(early_window).values - np.abs(late_window).values
            if np.all(diff>=0):
                windows_df.loc[windows_df.Station_Name==a_station, "win_size_" + str(a_window)]=a_row

windows_df = pd.merge(windows_df, 
                      DayRule28_no_snow_first_DoY[["Station_Name", "no_snow_first_DoY"]], 
                      on=['Station_Name'], how='left')
windows_df['DoY'] = 666

for a_idx in windows_df.index:
    windows_df.loc[a_idx, "DoY"] = windows_df.loc[a_idx, "no_snow_first_DoY"].day_of_year

windows_df    

# %%

# %%

# %%

# %%

# %%
