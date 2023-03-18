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
# import NASA_plot_core.py as rcp

# %%
dta_dir = "/Users/hn/Documents/01_research_data/RangeLand/"

# %%
nd = pd.read_csv(dta_dir+"qualityMosaicAttempt1_test_scale_30.csv")

# %%
nd.dropna(subset=['nd'], inplace=True)
nd.reset_index(drop=True, inplace=True)

# %%
nd["ID"]=nd["lat"].astype(str)+ "_" + nd["long"].astype(str) 

# %%
nd = nc.add_human_start_time_by_system_start_time(nd)

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 4), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
ax.grid(True);

plt.plot(nd[nd.ID==ID_1].month.values,
         nd[nd.ID==ID_1].nd.values,
         c='dodgerblue', linewidth=5);

ax.set_ylabel("Monthly NDVI")
ax.set_ylabel("Time")
ax.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
ax.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
# ax.legend(loc="upper right");

# ax.xaxis.set_major_locator(mdates.YearLocator(1))

ax_y_m = -1.1
ax_y_M=1.1
plt.yticks(np.arange(ax_y_m, ax_y_M, 0.4))
ax.set_ylim(ax_y_m, ax_y_M)


# %%

# %%
