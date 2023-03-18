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
# import warnings
# warnings.filterwarnings("ignore")

import csv
import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import scipy
import scipy.signal
import os, os.path

from datetime import date
import datetime
import time

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

# from pprint import pprint
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sb

import sys
import io

# to move files from one directory to another
import shutil


import yfinance as yf
from nasdaq_stock import nasdaq_stock as nasdaq_stock
import requests


from pylab import rcParams



# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('../')
import cycles_core as cc
import cycles_plot_core as cpc

# %%
data_dir = "/Users/hn/Documents/01_research_data/Supriya_challenge/Gridmet/"

# %%
# List all the csv files in the data directory
file_names = [x for x in os.listdir(data_dir) if x.endswith(".csv")]
file_names.sort()
file_names[0:2]

# %%

# %%
file_names[0:2]

# %%
###
###   Read CSV file. skip the first 16 rows
###
a_df = pd.read_csv(data_dir + file_names[0], skiprows=16)


###
###  sort by date column.
###
a_df.sort_values(by=['yyyy-mm-dd'], inplace=True)

###
###  Reset index
###
a_df.reset_index(drop=True, inplace=True)


a_df ['tmmx(C)'] = a_df['tmmx(K)'] - 273.15
a_df ['tmmn(C)'] = a_df['tmmn(K)'] - 273.15
a_df ['tavg(C)'] = (a_df['tmmx(C)'] + a_df['tmmn(C)']) / 2

# %%
print (a_df.shape)
a_df.head(3)

# %%
window_size = 10
a_df.rolling(window_size)

# %%
for window in a_df.rolling(window = 10):
    print(window)

# %%
