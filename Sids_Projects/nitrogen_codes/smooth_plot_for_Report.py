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

# %%
data_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/"

out_name = data_dir + "corn_potatoEq2_smoothed.csv"
corn_potato = pd.read_csv(out_name)

# %%
corn_potato.head(2)

# %%
corn_potato['human_system_start_time'] = pd.to_datetime(corn_potato['human_system_start_time'])

# %%
a_field = corn_potato.ID.unique()[-1]

fig, ax2 = plt.subplots(1, 1, figsize=(10, 3), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time,
         corn_potato[corn_potato.ID==a_field].nit,
         linewidth = 4, ls = '-', label = 'noisy signal', c="dodgerblue");

ax2.plot(corn_potato[corn_potato.ID==a_field].human_system_start_time,
         corn_potato[corn_potato.ID==a_field].smooth_window3,
         linewidth=4, ls='-', label='WMA-3', c="k");
# ax2.plot(corn_potato[corn_potato.ID==a_field].smooth_window5, linewidth=3, ls='-', label='WMA-5', c="r");

ax2.tick_params(axis='y', which='major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%Y-%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

plot_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/report_plots/"
os.makedirs(plot_dir, exist_ok=True)
file_name = plot_dir + "wma_no5.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + "wma_no5.png"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);


# %%
VI_idx = "V"
smooth_type = "N"
SR = 3
print (f"Passed Args. are: {VI_idx=:}, {smooth_type=:}, and {SR=:}!")

# %%
