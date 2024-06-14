# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import time, datetime
import sys, os, os.path
import scipy, scipy.signal

from datetime import date, datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score, \
                            confusion_matrix, balanced_accuracy_score, \
                            classification_report
import matplotlib.pyplot as plt

# from patsy import cr
# from pprint import pprint
# from statsmodels.sandbox.regression.predstd import wls_prediction_std

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp

# %%
NASA_dir_base = "/Users/hn/Documents/01_research_data/NASA/"
meta_dir = NASA_dir_base + "0000_parameters/"
train_TS_dir_base = NASA_dir_base + "VI_TS/"

data_part_of_shapefile_dir = NASA_dir_base + "data_part_of_shapefile/"


# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
ground_truth_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%
csv_data_parts = os.listdir(data_part_of_shapefile_dir)
csv_data_parts = [x for x in csv_data_parts if x.endswith("csv")]
csv_data_parts

# %%
irriigated_SF_data_concatenated = pd.read_csv(data_part_of_shapefile_dir + "irriigated_SF_data_concatenated.csv")
irriigated_SF_data_concatenated.head(2)

# %%
irriigated_SF_data_concatenated.county.unique()

# %%
irriigated_SF_data_concatenated["survey_year"] = pd.to_datetime(irriigated_SF_data_concatenated["LstSrvD"], 
                                                                format='mixed').dt.year

irriigated_SF_data_concatenated.head(2)

# %%
irriigated_SF_data_concatenated.head(3)

# %%
irriigated_SF_data_concatenated["SF_year"] = irriigated_SF_data_concatenated["ID"].str.split("_", expand=True)[3]
irriigated_SF_data_concatenated["SF_year"] = irriigated_SF_data_concatenated["SF_year"].astype(int)
irriigated_SF_data_concatenated.head(2)

# %%
(irriigated_SF_data_concatenated["SF_year"] == irriigated_SF_data_concatenated["survey_year"]).sum()

# %%
irriigated_SF_data_concatenated.shape

# %%
# limit to large fields
irriigated_SF_data_concatenated = irriigated_SF_data_concatenated[irriigated_SF_data_concatenated.ExctAcr > 10]

# %%
irriigated_SF_data_concatenated.head(2)

# %%
len(list(irriigated_SF_data_concatenated.CropTyp.unique()))

# %%
