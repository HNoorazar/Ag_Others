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
import warnings
warnings.filterwarnings("ignore")

import pickle, pandas as pd
from datetime import datetime
import os

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/Ehsan/wheat/"

# %%
# # !pip3 install openpyxl

# %%
merged_varieties = pd.read_excel(data_dir_base + "merged_varieties.xlsx")
merged_with_vars = pd.read_excel(data_dir_base + "merged_with_vars.xlsx")

# %%
merged_varieties.drop(["Location", "Year"], axis="columns", inplace=True)
merged_varieties.head(2)

# %%
list(merged_varieties.columns)

# %%

# %%
wheat_reOrganized = data_dir_base + "wheat_reOrganized/"
os.makedirs(wheat_reOrganized, exist_ok=True)

# %%
filename = wheat_reOrganized + "average_and_seperate_varieties.sav"

export_ = {"averaged_varieties": merged_varieties, 
           "separate_varieties": merged_with_vars, 
           "source_code" : "read_excel_dump",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
