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

# %% [markdown]
# Names are meaningless and useless data are exported. Clean them

# %%
import pandas as pd


from datetime import date
import sys, os, os.path, pickle, time

import matplotlib
import matplotlib.pyplot as plt


sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
data_base = "/Users/hn/Documents/01_research_data/Amin/Joel/Drive-by/"
in_dir = data_base + "data-to-plot/"
out_dir = data_base + "data_to_plot_cleaned/"
os.makedirs(out_dir, exist_ok=True)

# %%
in_csv_files = [x for x in os.listdir(in_dir) if x.endswith(".csv")]
print (len(in_csv_files))
in_csv_files[:2]

# %%
df = pd.read_csv(in_dir + in_csv_files[0])
df.head(2)

# %%
df = df[["ID", "NDVI", "system_start_time"]]
df = nc.add_human_start_time_by_system_start_time(df)
df.reset_index(drop=True, inplace=True)
df.head(2)

# %%
df['human_system_start_time'].max()

# %%
IDs = df["ID"].unique()
VI_idx = "NDVI"

# %%
a_field = df[df.ID == IDs[10]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

fig, ax = plt.subplots(1, 1, figsize=(12, 3), sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
                linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8)

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %%
