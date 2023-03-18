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
from scipy import fft
import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

# %%
snow_dir = "/Users/hn/Documents/01_research_data/Bhupi/snow/00/"
PMW_diff_dir = snow_dir + "PMW_difference_data/"
SNOTEL_dir = snow_dir + "SNOTEL_data/"

# %%
SNOTEL_Snow_depth_2013 = pd.read_csv(SNOTEL_dir + "SNOTEL_Snow_depth_2013.csv")
PMW_goodShape = pd.read_csv(PMW_diff_dir + "PMW_difference_data_2013_goodShape.csv")

print (SNOTEL_Snow_depth_2013.shape)
print (PMW_goodShape.shape)

# %%
DayRule28_no_snow_first_DoY=pd.read_csv(snow_dir + "DayRule28_no_snow_first_DoY.csv")
DayRule28_no_snow_first_DoY.head(2)

# %%
Station_Name = DayRule28_no_snow_first_DoY.Station_Name[0]
Station_Name

# %%
PMW_goodShape.head(2)

# %% [markdown]
# ### Weighted moving average

# %%
data=pd.DataFrame()
data['a_signal']=a_signal

window = 20
weights = np.arange(1, window+1)
wma = data['a_signal'].rolling(window).apply(lambda a_signal: np.dot(a_signal, weights)/weights.sum(), raw=True)

fig, ax2 = plt.subplots(1, 1, figsize=(10, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(a_signal, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
ax2.plot(wma, linewidth = 3, ls = '-', label = 'WMA', c="k");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

# %%
data=pd.DataFrame()
data['a_signal']=a_signal

window = 10
weights = np.arange(1, window+1)
wma = data['a_signal'].rolling(window).apply(lambda a_signal: np.dot(a_signal, weights)/weights.sum(), raw=True)

fig, ax2 = plt.subplots(1, 1, figsize=(10, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(a_signal, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
ax2.plot(wma, linewidth = 3, ls = '-', label = 'convolved', c="k");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

# %%
PMW_difference_data_2013.head(2)

# %%

# %%
