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
import shutup
shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
from datetime import datetime, date

# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# %% [markdown]
# ### Directories

# %%
wheat_database = "/Users/hn/Documents/01_research_data/Ehsan/wheat/"
hack_day_dir = wheat_database + "data_April_26_2025_hackday/"
separate_varieties_dir = hack_day_dir + "varieties/"


# %%
###### List of variety names
csv_files = [x for x in os.listdir(separate_varieties_dir) if x.endswith(".xlsx")]

# %%
variety_names = [s.split(".")[0] for s in csv_files]
variety_names[:3]

# %%
# all_varieties_grain_yields = pd.DataFrame()
# for a_file in csv_files:
#     df_ = pd.read_excel(separate_varieties_dir + a_file)
#     df_["variety"] = a_file.split(".")[0].lower()
#     all_varieties_grain_yields = pd.concat([all_varieties_grain_yields, df_])
    
# # change column names to lower case for consistency
# all_varieties_grain_yields.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
# all_varieties_grain_yields.head(2)

# filename = (separate_varieties_dir + "all_varieties_grainYields_stageAggregated.sav")

# export_ = {
#     "all_varieties_grain_yields": all_varieties_grain_yields,
#     "source_code": "corr_feat_select_separate_varieties",
#     "Author": "HN",
#     "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# }

# pickle.dump(export_, open(filename, "wb"))

# all_varieties_grain_yields.head(2)

# %%
filename = separate_varieties_dir + "all_varieties_grainYields_stageAggregated.sav"
all_varieties_GY = pd.read_pickle(filename)
all_varieties_GY = all_varieties_GY["all_varieties_grain_yields"]
all_varieties_GY.head(2)

# %%
variety_names = list(all_varieties_GY["variety"].unique())
variety_names[0:3]

# %%
# Model for a given variety and later we do a for-loop or Kamiak parallel

a_variety = variety_names[0]
df_variety = all_varieties_GY[all_varieties_GY["variety"] == a_variety].copy()
df_variety.reset_index(drop=True, inplace=True)
df_variety.head(3)

# %%
list(df_variety.columns)

# %%
df_variety.head(2)

# %%
variable = "elevation"
a_years_data = df_variety[df_variety.year== 2008]
a_years_data[["location", variable]].groupby([variable]).count()

# %%

# %%
# df_variety[["location", "soil_type"]].groupby(["soil_type"]).count()

# %%
# df_variety[["location", "slope"]].groupby(["slope"]).count()

# %%
all_columns = sorted(list(df_variety.columns))
print (len(all_columns))
all_columns[0:3]

# %%
# some of these i do not like and some are constant (e.g. latitude). 
# what is the meaning of latitude's correlation with precipitation that changes over time?
# I donno
unwanted_features = ["year", "slope", "soil_type", 
                     "location", "latitude", "longitude", 
                     "aspect", "elevation", "variety"]
wanted_features = [x for x in all_columns if not (x in unwanted_features)]
wanted_features

# %%

# %%
traits = ["grain_yield"] # , "height", "protein"

# %%

len(a)

# %%
df_variety["variety"].unique()

# %%
all_outputs = {}

for features_corr_thresh in [0.5, .6, .7, .8]:
    for a_variety in variety_names:
        df_variety = all_varieties_GY[all_varieties_GY["variety"] == a_variety].copy()

        # Perform analysis for each trait
        for trait in traits:
            df = df_variety[wanted_features].copy() # Drop unwanted columns

            # Remove the other three traits (keep only the target trait)
            traits_to_remove = [t for t in traits if t != trait]
            df.drop(columns=traits_to_remove, inplace=True)

            # the rest of the code assumes target variable is at the end of DF
            df[trait] = df.pop(trait)

            # Compute the correlation matrix
            correlation_matrix = df.corr()

            # Identify pairs of highly correlated variables (correlation > features_corr_thresh)
            high_corr_pairs = []
            for i, col1 in enumerate(df.columns[:-1]): # Exclude the target column
                for col2 in df.columns[i+1:-1]:
                    if abs(correlation_matrix.loc[col1, col2]) > features_corr_thresh:
                        high_corr_pairs.append((col1, col2))

            # Decide which variable to remove based on correlation with target trait
            removed_features = set()
            for col1, col2 in high_corr_pairs:
                corr1 = abs(correlation_matrix.loc[col1, trait])
                corr2 = abs(correlation_matrix.loc[col2, trait])
                if corr1 < corr2:
                    removed_features.add(col1)
                else:
                    removed_features.add(col2)

            # Filter the DataFrame to remove the highly correlated variables
            filtered_df = df.drop(columns=list(removed_features))

            # Add year location back to the filtered DataFrame
            filtered_df = pd.concat([df_variety[unwanted_features], filtered_df], axis=1)

            filtered_df['environment'] = filtered_df['location'].astype(str) + '_' + \
                                                    filtered_df['year'].astype(str)
            
            key_ = f"{trait}_{a_variety}_corrFeatSelect_Thresh{int(features_corr_thresh*10)}"
            all_outputs[key_] = filtered_df
            
filename = separate_varieties_dir + f"{trait}_corrFeatSelect.sav"
export_ = {"all_varieties_differentThreshs": all_outputs,
           "source_code": "corr_feat_select_separate_varieties",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))


# %%
filename = separate_varieties_dir + f"{trait}_corrFeatSelect_Thresh{int(features_corr_thresh*10)}.sav"
grain_yield_corrFeature = pd.read_pickle(filename)
grain_yield_corrFeature = grain_yield_corrFeature["filtered_df"]
grain_yield_corrFeature.head(2)

# %%
grain_yield_corrFeature.shape

# %%
trait

# %%
del(a)

# %%

# %%
#----------------------------#
#        HEADING DATE        #
#----------------------------#
# perform pairwise correlation analysis
df_env = pd.read_csv('df_22805_hd.csv')
df = df_env.drop(columns=['location', 'year'])

# Compute the correlation matrix
correlation_matrix = df.corr()

# Identify pairs of highly correlated variables (correlation > 0.6)
high_corr_pairs = []
for i, col1 in enumerate(df.columns[:-1]):  # Exclude the target column
    for col2 in df.columns[i+1:-1]:
        if abs(correlation_matrix.loc[col1, col2]) > 0.7:
            high_corr_pairs.append((col1, col2))

# Decide which variable to remove based on correlation with target
removed_features = set()
for col1, col2 in high_corr_pairs:
    corr1 = abs(correlation_matrix.loc[col1, 'Heading_date'])
    corr2 = abs(correlation_matrix.loc[col2, 'Heading_date'])
    if corr1 < corr2:
        removed_features.add(col1)
    else:
        removed_features.add(col2)

# Filter the DataFrame to remove the highly correlated variables
filtered_df = df.drop(columns=list(removed_features))
# Add year location back to the filtered DataFrame
filtered_df = pd.concat([df_env[drop_columns], filtered_df], axis=1)
# Add a new column named environment by combining location and year
filtered_df['environment'] = filtered_df['location'].astype(str) + '_' + filtered_df['year'].astype(str)
# filtered_df.to_csv('hd_33125.csv', index  = False)

removed_features
