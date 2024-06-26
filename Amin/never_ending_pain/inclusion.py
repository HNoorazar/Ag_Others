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
out_name = "/Users/hn/Documents/01_research_data/NASA/" + "all_fields_correct_year_irr_noNass.csv"
pool = pd.read_csv(out_name)
pool.head(2)

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
GT_wMeta = pd.merge(ground_truth_labels, meta, on="ID", how="left")
print (GT_wMeta.shape)
GT_wMeta.head(2)

# %%
GT_crops = GT_wMeta["CropTyp"].unique()
meta_crops = meta["CropTyp"].unique()

print (len(GT_crops))
print (len(meta_crops))

# %%
[x for x in meta_crops if not(x in GT_crops)]

# %%
GT_wMeta[GT_wMeta["CropTyp"] == "unknown"].shape

# %%
# csv_data_parts = os.listdir(data_part_of_shapefile_dir)
# csv_data_parts = [x for x in csv_data_parts if x.endswith("csv")]
# csv_data_parts

# %%
# irriigated_SF_data_concatenated = pd.read_csv(data_part_of_shapefile_dir + "irriigated_SF_data_concatenated.csv")
# irriigated_SF_data_concatenated["survey_year"] = pd.to_datetime(irriigated_SF_data_concatenated["LstSrvD"], 
#                                                                 format='mixed').dt.year

# print (irriigated_SF_data_concatenated.county.unique())

# irriigated_SF_data_concatenated["SF_year"] = irriigated_SF_data_concatenated["ID"].str.split("_", expand=True)[3]
# irriigated_SF_data_concatenated["SF_year"] = irriigated_SF_data_concatenated["SF_year"].astype(int)
# irriigated_SF_data_concatenated.head(2)

# %%
# (irriigated_SF_data_concatenated["SF_year"] == irriigated_SF_data_concatenated["survey_year"]).sum()

# print (irriigated_SF_data_concatenated.shape)
# # limit to large fields
# irriigated_SF_data_concatenated = irriigated_SF_data_concatenated[irriigated_SF_data_concatenated.ExctAcr > 10]
# print (len(list(irriigated_SF_data_concatenated.CropTyp.unique())))
# irriigated_SF_data_concatenated.head(2)

# %%
print (len(pool.CropTyp.unique()))
print (len(GT_wMeta.CropTyp.unique()))

# %%
# pool = pool[pool.ExctAcr > 10].copy()
pool = pool[pool.CropTyp != "unknown"]
pool = pool[pool.CropTyp.isin(list(GT_wMeta.CropTyp.unique()))]

# %%
GT_wMeta = GT_wMeta[GT_wMeta.CropTyp != "unknown"]
GT_wMeta.head(2)

# %%
numer = pd.DataFrame(GT_wMeta.groupby(["CropTyp"])["ID"].count()).reset_index()
numer.rename(columns={"ID": "numer"}, inplace=True)
numer.head(2)

# %%
denom = pd.DataFrame(pool.groupby(["CropTyp"])["ID"].count()).reset_index()
denom.rename(columns={"ID": "denom"}, inplace=True)
denom.head(2)

# %%
in_df = pd.DataFrame({"CropTyp" : sorted(pool.CropTyp.unique())})
in_df.head(2)

# %%
print (in_df.shape)
in_df = pd.merge(in_df, numer, on="CropTyp", how="outer")
print (in_df.shape)
in_df = pd.merge(in_df, denom, on="CropTyp", how="outer")
print (in_df.shape)
in_df.head(2)

# %%
in_df["inclusion_prob"] = in_df["numer"] / in_df["denom"]

# %%
in_df.head(2)

# %%
in_df.head(2)

# %%
pool.head(2)

# %%
sorted(list(pool.Irrigtn.unique()))

# %%
sorted(list(pool.DataSrc.unique()))

# %%
pool.county.unique()

# %%
pool["SF_year"] = pool["ID"].str.split("_", expand=True)[3]
pool["SF_year"] = pool["SF_year"].astype(int)

pool_Adams = pool[pool.county == "Adams"].copy()
pool_Benton = pool[pool.county == "Benton"].copy()
pool_Franklin = pool[pool.county == "Franklin"].copy()
pool_Yakima = pool[pool.county == "Yakima"].copy()
pool_Grant = pool[pool.county == "Grant"].copy()
pool_Walla = pool[pool.county == "Walla Walla"].copy()

# %%
print (pool_Adams.SF_year.unique())
print (pool_Benton.SF_year.unique())
print (pool_Franklin.SF_year.unique())
print (pool_Yakima.SF_year.unique())
print (pool_Grant.SF_year.unique())
print (pool_Walla.SF_year.unique())

# %%
in_df = in_df.round(3)

out_name = "/Users/hn/Documents/01_research_data/Amin/inclusion.csv"
in_df.to_csv(out_name, index = False)

# %%
A = ["Phenological patterns  (start, end, peak) ??",
"Temperature",
"Precipitation",
"Radiation",
"RH",
"Potential ET",
"Actual ET - MODIS ET",
"Soil moisture VIC",
"GDD (0-degree baseline, no upper threshold)  Try multiple options to check for correlation",
"HDD (multiple thresholds)",
"dGDD",
"DTR",
"Drought index SPEI  12 month (monthly)",
"Day length",
"Soil characteristics",
"Vegetation type"]

sorted(A)

# %%

# %%
