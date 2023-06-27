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
import pandas as pd

# %%
tim = pd.read_csv("/Users/hn/Documents/01_research_data/Ehsan/tim_gdd_PHHDates.csv")
tim.head(2)

# %%
cols = ["planting_date", "heading_date", "harvest_date"]

# %%
tim.location = tim.location.astype(str)
tim.heading_date = tim.heading_date.astype(str)
tim.harvest_date = tim.harvest_date.astype(str)
tim.planting_date = tim.planting_date.astype(str)
tim.year = tim.year.astype(str)

# %%
for a_col in cols:
    for row in tim.index:
        if (len(tim.loc[row, a_col].split("-")) > 1):
            tim.loc[row, a_col] = ",".join(tim.loc[row, a_col].split("-"))
tim.head(2)


# %%
for row in tim.index:
        if (len(tim.loc[row, "year"].split(".")) > 1):
            tim.loc[row, "year"] = tim.year[0].split(".")[0]

# %%
import numpy as np
tim.replace("nan", np.nan, inplace=True)

# %%
tim.head(20)

# %%
out_name = "/Users/hn/Documents/01_research_data/Ehsan/tim_gdd_PHHDates.csv"
tim.to_csv(out_name, index = False)

# %%

# %%
