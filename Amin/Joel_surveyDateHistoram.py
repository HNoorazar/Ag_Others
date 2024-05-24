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

# %%
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow
import sys, os, os.path, pickle, time

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
data_dir_ = "/Users/hn/Documents/01_research_data/Amin/Joel/"
plot_dir = data_dir_ + "plots/"

os.makedirs(plot_dir, exist_ok=True)

# %%
data_2022_nofilter = pd.read_csv(data_dir_ + "data_2022_nofilter.csv")
data_2023_nofilter = pd.read_csv(data_dir_ + "data_2023_nofilter.csv")

print (data_2022_nofilter.shape)
print (data_2023_nofilter.shape)

# %%
data_2022_nofilter.head(2)

# %%
data_2022_nofilter.LstSrvD.max()

# %%

# %%
data_2022_nofilter = pd.read_csv(data_dir_ + "data_2022_nofilter.csv")
data_2022_nofilter["CropTyp"] = data_2022_nofilter["CropTyp"].str.lower()
data_2022_nofilter.drop(columns=["Unnamed: 0"], inplace=True)

### Rename column names: lower case for consistency
data_2022_nofilter.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

##### Sort by id
data_2022_nofilter.sort_values(by=["id"], inplace=True)
data_2022_nofilter.reset_index(drop=True, inplace=True)

data_2022_nofilter.head(2)

# %%
### Convert type of lstsrvd from string to date
data_2022_nofilter.lstsrvd = pd.to_datetime(data_2022_nofilter.lstsrvd)
data_2022_nofilter["last_survey_year"] = data_2022_nofilter.lstsrvd.dt.year
data_2022_nofilter.head(2)

# %%
tick_legend_FontSize = 10

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
df = data_2022_nofilter.copy()
fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, gridspec_kw={"hspace": 0.35, "wspace": 0.05})
axs.grid(axis="y", which="both")

LL = len(df["last_survey_year"].unique())
X_axis = np.arange(LL)

bar_width_ = 1
step_size_ = 5 * bar_width_
X_axis = np.array(range(0, step_size_ * LL, step_size_))

# axs.bar(X_axis - bar_width_ * 2, df[["id", "last_survey_year"]], 
#         color="dodgerblue",
#         width=bar_width_)

# %%
df[df["last_survey_year"] == df["last_survey_year"].unique().max()]

# %%
