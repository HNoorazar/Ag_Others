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

# %%
##
## Read data
##
potato = pd.read_csv(data_dir + "Potato_Sent_Sept28_2ndFormula_2020-01-01_2021-01-01.csv")
corn = pd.read_csv(data_dir + "Corn_Potato_Sent_2020-01-01_2021-01-01.csv")
metadata = pd.read_csv(data_dir + "corn_potato_metadata2020.csv")

# %%
## Toss small fields
metadata=metadata[metadata.ExactAcres>10]

# %%
print (metadata.shape)
metadata=metadata[metadata.CropTyp.isin(["Corn, Field", "Potato"])]
print (metadata.shape)

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

# Drop NAs in potato
print (potato.shape)
potato.dropna(subset=["NuptakeGEE"], axis=0, inplace=True)
print (potato.shape)

potato.reset_index(drop=True, inplace=True)
potato.head(2)

### Compute Corn's uptake
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


corn["chl"]=corn["CIRed"]*6.68
corn["chl"]=corn["chl"]-0.67
corn["nit"] = corn["chl"]*4.73+0.27

corn.head(3)


potato.rename(columns={'NuptakeGEE': 'nit'}, inplace=True)

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

# %%
out_name = data_dir + "corn_potatoEq2_smoothed.csv"
corn_potatoEq2_smoothed = pd.read_csv(out_name)

# %%
corn_potatoEq2_smoothed.groupby('CropTyp').agg('min')

# %%
corn_potatoEq2_smoothed.groupby('CropTyp').agg('max')

# %%
corn.sort_values("nit", ascending=False).head(10)

# %%
corn_potatoEq2_smoothed

# %%
meta_data = pd.read_csv("/Users/hn/Documents/01_research_data/Sid/Nitrogen_data/corn_potato_metadata2020.csv")

# %%
meta_data.head(2)

# %%
extended = pd.merge(corn_potatoEq2_smoothed, meta_data[["ID", "Acres"]], on=['ID'], how='left')
extended.head(2)

# %%
extended.Acres.min()

# %%
extended.groupby('CropTyp').agg('sum')

# %%
extended.groupby(['CropTyp'])['CropTyp'].count()

# %%
