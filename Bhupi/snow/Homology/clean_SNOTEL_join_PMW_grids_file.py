# ---
# jupyter:
#   jupytext:
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

from datetime import date, datetime
import time

import random
from random import seed
from random import random

import os, os.path
import shutil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from pylab import imshow
import pickle
import h5py
import sys

# %%
snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EithyYearsClustering/"
# diff_dir = snow_TS_dir_base+ "Brightness_temperature/Only_for_SNOTEL_grids/"
SNOTEL_dir = snow_TS_dir_base + "SNOTEL_observations/"

# %%
SNOTEL_join_PMW_grids = pd.read_csv(SNOTEL_dir + "SNOTEL_join_PMW_grids.csv")

new_cols = [a_col.replace(".", "_") for a_col in list(SNOTEL_join_PMW_grids.columns)]
SNOTEL_join_PMW_grids.columns = new_cols

new_cols = [a_col.lower() for a_col in list(SNOTEL_join_PMW_grids.columns)]
SNOTEL_join_PMW_grids.columns = new_cols

SNOTEL_join_PMW_grids.to_csv(SNOTEL_dir + "SNOTEL_join_PMW_grids.csv" , index=False)

# %%

# %%

# %%

# %%
