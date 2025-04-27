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

from sklearn.model_selection import train_test_split

# %%
wheat_database = "/Users/hn/Documents/01_research_data/Ehsan/wheat/"
hack_day_dir = wheat_database + "data_April_26_2025_hackday/"
separate_varieties_dir = hack_day_dir + "varieties/"

# %%
grain_yield

# %%
trait = "grain_yield"

# %%
filename = separate_varieties_dir + f"{trait}_corrFeatSelect.sav"
grain_yield_corrFeature = pd.read_pickle(filename)
grain_yield_corrFeature = grain_yield_corrFeature["all_varieties_differentThreshs"]
grain_yield_corrFeature.keys()

# %%
list (grain_yield_corrFeature.keys())

# %%

# %%
# grain_yield_corrFeature.drop(["environment"], axis=1, inplace=True)
all_columns = list(grain_yield_corrFeature.columns);

unwanted_features = ["year", "slope", "soil_type", 
                     "location", "latitude", "longitude", 
                     "aspect", "elevation", "variety"]
wanted_features = [x for x in all_columns if not (x in unwanted_features)]

grain_yield_corrFeature = grain_yield_corrFeature[wanted_features]

# %%
y_var = "grain_yield"
print (len(wanted_features))
wanted_features.remove(y_var)
print (len(wanted_features))

# %%
grain_yield_corrFeature[]
variety = 

# %%
x_train, x_test, y_train, y_test = train_test_split(grain_yield_corrFeature[wanted_features], 
                                                    grain_yield_corrFeature[y_var], 
                                                    test_size=0.2, random_state=0, shuffle=True)

# %%
parameters = {'n_jobs':[6],
              'criterion': ["squared_error", "friedman_mse"], 
              'max_depth':[1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17],
              'min_samples_split':[4],
              'max_features': ["log2"],
              'ccp_alpha':[0.0, 0.1, 0.2, ], 
              'max_samples':[None]
             } # , 
regular_forest_grid_1 = GridSearchCV(RandomForestRegressor(random_state=0), 
                                     parameters, cv=5, verbose=1,
                                     error_score='raise')

regular_forest_grid_1.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (regular_forest_grid_1.best_params_)
print (regular_forest_grid_1.best_score_)

# %%
regular_forest_grid_1_predictions = regular_forest_grid_1.predict(x_test_df.iloc[:, 1:])
regular_forest_grid_1_y_test_df=y_test_df.copy()
regular_forest_grid_1_y_test_df["prediction"]=list(regular_forest_grid_1_predictions)
regular_forest_grid_1_y_test_df.head(2)
