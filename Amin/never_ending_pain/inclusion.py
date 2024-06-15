# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
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
train_TS_dir_base = NASA_dir_base + "VI_TS/"
training_set_dir = NASA_dir_base + "/ML_data_Oct17/"

meta_dir = NASA_dir_base + "parameters/"
data_part_of_shapefile_dir = NASA_dir_base + "data_part_of_shapefile/"


# %%
# MacBook directories

NASA_dir_base = "/Users/hn/Documents/01_research_data/Amin/inclusion_prob/"
data_part_of_shapefile_dir = NASA_dir_base + "data_part_of_shapefile/"
training_set_dir = NASA_dir_base

# %%

# meta = pd.read_csv(meta_dir+"evaluation_set.csv")
# meta_moreThan10Acr=meta[meta.ExctAcr > 10]
# print (meta.shape)
# print (meta_moreThan10Acr.shape)
# meta.head(2)

# %%
ground_truth_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%
csv_data_parts = os.listdir(data_part_of_shapefile_dir)
csv_data_parts = [x for x in csv_data_parts if x.endswith("csv")]
csv_data_parts

# %%
irriigated_SF_data = pd.read_csv(data_part_of_shapefile_dir + "irriigated_SF_data_concatenated.csv")
irriigated_SF_data.head(2)

# %%
irriigated_SF_data.county.unique()

# %%
irriigated_SF_data["survey_year"] = pd.to_datetime(irriigated_SF_data["LstSrvD"], 
                                                                format='mixed').dt.year

irriigated_SF_data.head(2)

# %%
irriigated_SF_data.head(3)

# %%
irriigated_SF_data["SF_year"] = irriigated_SF_data["ID"].str.split("_", expand=True)[3]
irriigated_SF_data["SF_year"] = irriigated_SF_data["SF_year"].astype(int)
irriigated_SF_data.head(2)

# %%
(irriigated_SF_data["SF_year"] == irriigated_SF_data["survey_year"]).sum()

# %%
irriigated_SF_data.shape

# %%
# limit to large fields
# irriigated_SF_data = irriigated_SF_data[irriigated_SF_data.ExctAcr > 10]

# %%
irriigated_SF_data.head(2)

# %%
len(list(irriigated_SF_data.CropTyp.unique()))

# %%
ground_truth_labels.shape

# %%
all_SF_data = pd.read_csv(data_part_of_shapefile_dir + "all_SF_data_concatenated.csv")
all_SF_data.head(2)

# %%
all_SF_data.shape

# %%
ground_truth_labels = pd.merge(ground_truth_labels, all_SF_data, how="left", on="ID")
print (f"{ground_truth_labels.shape = }")
ground_truth_labels.head(2)

# %%
ground_truth_labels = ground_truth_labels[['ID', 'Vote', 'CropTyp',
                                           'Acres', 'Irrigtn', 'LstSrvD',
                                           'DataSrc', 'county']]

ground_truth_labels.shape

# %%
print (all_SF_data.county.unique())
print ()
print (all_SF_data.Irrigtn.unique())

# %%
GT_cropTypes = list(ground_truth_labels.CropTyp.unique())
irriigated_SF_data = irriigated_SF_data[irriigated_SF_data.CropTyp.isin(GT_cropTypes)].copy()

irriigated_SF_data.shape

# %%
irriigated_SF_data.ExctAcr.min()

# %%
all_SF_data.CropTyp = all_SF_data.CropTyp.str.lower()

# %%

# %%
## see which crop types are included in training process
## where all fields of that crop are included (not just 10%).

crops_100Percent_in_train = []
percentage_dict = {}

# for crop in GT_cropTypes:
#     curr_irr_all = irriigated_SF_data[irriigated_SF_data.CropTyp == crop].copy()
#     GT_fields = ground_truth_labels[ground_truth_labels.CropTyp == crop].copy()
#     if len(curr_irr_all) == len(GT_fields):
#         crops_100Percent_in_train += [crop]
        
#     percentage_dict[crop] = round( (len(GT_fields) / len(curr_irr_all)) * 100, 2)
    
    


for crop in GT_cropTypes:
    curr_irr_all = irriigated_SF_data[irriigated_SF_data.CropTyp == crop].copy()
    curr_irr_all_large = curr_irr_all[curr_irr_all.ExctAcr > 10 ]
    GT_fields = ground_truth_labels[ground_truth_labels.CropTyp == crop].copy()
    
    perc_ = round( (len(GT_fields) / len(curr_irr_all)) * 100, 2)
    perc_large_ = round( (len(GT_fields) / len(curr_irr_all_large)) * 100, 2)
    percentage_dict[crop] = [perc_, perc_large_]
    

# %%
ground_truth_labels.shape

# %%
print (irriigated_SF_data[irriigated_SF_data.CropTyp == "canola"].shape)
print (irriigated_SF_data[(irriigated_SF_data.CropTyp == "canola") & \
                         (irriigated_SF_data.ExctAcr > 10)].shape)

print (ground_truth_labels[ground_truth_labels.CropTyp == "canola"].shape)


# IF we look at irriigated_SF_data There should be 6 fields of canola.
# why there are 27? Look at evaluation set. And why evaluation set is larger than
# irriigated_SF_data? counties?

# %%

# %%
percentage_dict

# %%
    

# %%

# %%

# %%

# %%
# "ryegrass seed" and "medicinal herb" 
ground_truth_labels[ground_truth_labels.CropTyp == "ryegrass seed"]

# %%
list(ground_truth_labels.CropTyp.unique())

# %%
ground_truth_labels[ground_truth_labels.CropTyp == "wildlife feed"].shape

# %%
irriigated_SF_data.head(2)

# %%
irriigated_SF_data[irriigated_SF_data.CropTyp == "barley hay"].shape

# %%
ground_truth_labels[ground_truth_labels.CropTyp == "barley hay"].shape

# %%
irriigated_SF_data[(irriigated_SF_data.CropTyp == "barley hay") &\
                   (irriigated_SF_data.ExctAcr > 10)].shape

# %%

# %%
