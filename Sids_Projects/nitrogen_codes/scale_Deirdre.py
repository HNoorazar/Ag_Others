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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

# %%
data_dir = "/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/"

# %%
Deirdre = pd.read_csv(data_dir + "Deirdre/InorganicNdatasummarizedbyproduct_20221209.csv")

# %%

# %%
scaled_dt =Deirdre.copy()
scaled_dt.head(2)

# %%
list(scaled_dt.columns)

# %%
for an_ID in list(scaled_dt["Product.Type"].unique()):
    curr_field = scaled_dt[scaled_dt["Product.Type"]==an_ID]
    scaler = MinMaxScaler()
    aa = scaler.fit_transform(curr_field[['Inorganic N (g/m2) mean']]).reshape(-1)
    scaled_dt.loc[curr_field.index, "Inorganic N (g/m2) mean"]=aa
    
    aa = scaler.fit_transform(curr_field[['Mineralized Inorganic N (g/m2)']]).reshape(-1)
    scaled_dt.loc[curr_field.index, "Mineralized Inorganic N (g/m2)"]=aa

# %%
out_name = data_dir + "Deirdre/03_scaled_Deirdre.csv"
scaled_dt.to_csv(out_name, index = False)

# %%
