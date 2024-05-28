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
years = np.arange(2021, 2024)

all_no_filter = pd.DataFrame()

for year in years:
    data_nofilter = pd.read_csv(data_dir_ + "To_Joel/" + "data_" + str(year) + "_nofilter.csv")
    print (data_nofilter.shape)

    data_nofilter["CropTyp"] = data_nofilter["CropTyp"].str.lower()

    data_nofilter.drop(columns=["Unnamed: 0"], inplace=True)

    ### Rename column names: lower case for consistency
    data_nofilter.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

    ##### Sort by id
    data_nofilter.sort_values(by=["id"], inplace=True)
    data_nofilter.reset_index(drop=True, inplace=True)
    
    data_nofilter.lstsrvd = pd.to_datetime(data_nofilter.lstsrvd)
    
    
    data_nofilter["last_survey_year"] = data_nofilter.lstsrvd.dt.year
    data_nofilter["image_year"] = year

    all_no_filter = pd.concat([all_no_filter, data_nofilter])
    
    
    all_no_filter["correct_year"] = False
    all_no_filter.loc[
        all_no_filter.last_survey_year == all_no_filter.image_year, "correct_year"] = True
    
all_no_filter.head(2)

# %%
all_correct_year = all_no_filter[all_no_filter.correct_year == True].copy()

all_correct_year.drop(columns=["correct_year"], inplace=True)
all_correct_year.sort_values(by=["id"], inplace=True)
all_correct_year.reset_index(drop=True, inplace=True)

print (all_correct_year.shape)
all_correct_year.head(2)

# %%
all_no_filter.to_csv(data_dir_ + "To_Joel/" + "all_no_filter_2021_to_2023.csv", index=False)
all_correct_year.to_csv(data_dir_ + "To_Joel/" + "all_correct_year_2021_to_2023.csv", index=False)

# %%
all_no_filter.head(2)

# %%
all_no_filter_wide = all_no_filter[["id", "label", "image_year"]].copy()
all_no_filter_wide["image_year"] = all_no_filter_wide["image_year"].astype("str")
all_no_filter_wide = all_no_filter_wide.pivot(index="id", columns = "image_year", values = "label").reset_index()
all_no_filter_wide.columns = all_no_filter_wide.columns.values
all_no_filter_wide.rename(columns={"2021": "label_2021",
                                   "2022": "label_2022",
                                   "2023": "label_2023"},
                         inplace=True)

needed_cols = ["id", 'acres', 'county', 'croptyp', 'datasrc', 'irrigtn', 'lstsrvd', 'last_survey_year']
all_no_filter_wide = pd.merge(all_no_filter_wide, all_correct_year[needed_cols], on=["id"], how="left")
all_no_filter_wide.head(2)
all_no_filter_wide.to_csv(data_dir_ + "To_Joel/" + "all_correct_year_2021_to_2023_wide.csv", index=False)

# %% [markdown]
# ## read shapefile

# %%
import folium, json
import geopandas as gpd

# %%
# %time
SF = gpd.read_file(data_dir_ + "From_Joel/" + "Joel_shp/WSDACrop_2023_WSUDoubleCrop.shp", crs='EPSG:4326')
SF.OBJECTID = SF.OBJECTID.astype(int)
SF.head(2)

# %%
all_correct_year.head(2)

# %%
needed_cols = ['id', 'prob_single', 'label', 'last_survey_year', 'image_year']
SF = pd.merge(SF, all_correct_year[needed_cols], 
              left_on=["OBJECTID"], right_on=["id"], how="left")
SF.drop(columns="id", inplace=True)
SF.head(2)

# %%

# %%

# %%
SF.to_file(filename = data_dir_ + "To_Joel/" + 'SF_2021_to_2023_DC_May272024.shp.zip', driver='ESRI Shapefile')

# %%
# %time
# WSDA2023DoubleCropOnly_May2024 = gpd.read_file(data_dir_ + "From_Joel/WSDA2023DoubleCropOnly_Mat2024/" + 
#                                                'WSDA2023DoubleCropOnly.shp', crs='EPSG:4326')
# WSDA2023DoubleCropOnly_May2024.head(2)

# %%
# weird_SF_dir = data_dir_ + "From_Joel/WSDACrop_2023_WSUDoubleCrop_V2/"

# mallard_0 = gpd.read_file(weird_SF_dir + "a0000000.gdb", layer=1)
# # mallard_1 = gpd.read_file(weird_SF_dir + "a0000000a.gdb", layer=1)

# %%

# %%
Whitman = SF[SF.County == "Whitman"].copy()
Whitman[Whitman.label == "double-cropped"]

# %% [markdown]
# # Export data for Jeol and Kirti

# %%

# %%
# Check w/ Kirti. Some of these are new to me. Is tea and kiwi produced here?!

bad_crops_2Drop = [
    "0",
    "CRP/Conservation",
    "Christmas Tree",
    "Cover Crop",
    "Dandelion",
    "Developed",
    "Driving Range",
    "Echinacea",
    #                   'Fallow',
    #                   'Fallow, Idle',
    #                   'Fallow, Tilled',
    "Golf Course",
    "Hemp",
    "Herb, Unknown",
    "Kiwi",
    #                   'Nursery, Caneberry',
    #                   'Nursery, Greenhouse',
    #                   'Nursery, Lavender',
    #                   'Nursery, Orchard/Vineyard',
    #                   'Nursery, Ornamental',
    "Peony",
    "Reclamation Seed",
    "Research Station",
    "Silviculture",
    "Sod Farm",
    "Tea",
    "Unknown",
    "Wildlife Feed",
]
bad_crops_2Drop = [x.lower() for x in bad_crops_2Drop]

# %%
all_no_filter = all_no_filter[~all_no_filter.croptyp.isin(bad_crops_2Drop)]
all_no_filter.reset_index(drop=True, inplace=True)
all_no_filter.head(2)

all_correct_year = all_correct_year[~all_correct_year.croptyp.isin(bad_crops_2Drop)]
all_correct_year.reset_index(drop=True, inplace=True)
all_correct_year.head(2)

# %%
all_correct_year.head(2)

# %%
print(f"{all_no_filter.shape = }")
print(f"{all_correct_year.shape = }")

# %% [markdown]
# # Create (at least) 4 tables
#
# Two tables for each year
#
# - One table say how many/acres are double cropped in general.
# - One table say how many/acres are double cropped using proper survey date so that they can dive in and see things based on crop-type and whatnot!!!
# - Extra tables can have crop types in it as well.

# %% [markdown]
# ### No-Filter tables

# %%
yr = 2021
print("total acre is [{}].".format(all_no_filter[all_no_filter["image_year"] == yr]["acres"].sum()))

# %%
pd.DataFrame(all_no_filter[all_no_filter["image_year"] == yr]\
             .groupby("label")["id"].count()).reset_index()

# %%
pd.DataFrame(all_no_filter[all_no_filter["image_year"] == yr].groupby("label")["acres"].sum()).reset_index()

# %%

# %% [markdown]
# ### No-Filter tables: counties

# %%
yr = 2021

df = all_no_filter[all_no_filter["image_year"] == yr].copy()
field_count_counties = pd.DataFrame(df.groupby(["county"])["id"].count()).reset_index()
field_count_counties.rename(columns={"id": "total_field_count"}, inplace=True)

field_acr_counties = pd.DataFrame(df.groupby(["county"])["acres"].sum()).reset_index()
field_acr_counties.rename(columns={"id": "acres"}, inplace=True)

county_field_countAcr = pd.merge(
    field_count_counties, field_acr_counties, on=["county"], how="left")
county_field_countAcr.head(2)

# %%

# %%
df = all_no_filter[all_no_filter["image_year"] == yr].copy()

data_nofilter_labelCounts = pd.DataFrame(df.groupby(["county", "label"])["id"].count()).reset_index()
data_nofilter_labelCounts.rename(columns={"id": "field_count"}, inplace=True)

data_nofilter_labelAcr = pd.DataFrame(df.groupby(["county", "label"])["acres"].sum()).reset_index()

county_nofilter_labelsCountAcr = pd.merge(data_nofilter_labelAcr, data_nofilter_labelCounts,
                                          on=["county", "label"], how="left")
county_nofilter_labelsCountAcr.head(2)

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

color_dict = {
    "single-cropped": "#DDCC77",
    "double-cropped": "#332288",
}

color_dict = {
    "single-cropped": "dodgerblue",
    "double-cropped": "red",
}

# %%
plot_col = "acres"
df = county_nofilter_labelsCountAcr.copy()
df = df.pivot(index="county", columns="label", values=plot_col).reset_index(drop=False)
df.columns = df.columns.values
df.plot(x="county", kind="bar", stacked=False)
# title=plot_col
plt.xlabel("county")
plt.ylabel(plot_col)

# %%

# %%
import plotly.express as px

df = county_nofilter_labelsCountAcr.copy()
fig = px.bar(
    df, x="county", y="acres", color="label", barmode="group", text="acres", height=400
)

fig.update_xaxes(categoryorder="array", categoryarray=df.county.unique())

# fig.update_layout(font=dict(# textfont_size=20,
#                             family="Courier New, monospace",
#                             size=18,  # Set the font size here
#                             color="RebeccaPurple")
#              )
# fig.update_traces(textfont_size=80)

# file_name = data_dir_ + "county_2022_nofilter_labelsAcr.pdf"
# fig.write_image(file_name) need to install kaleido
fig.show()

# %%

# %%
plot_col = "acres"
df = county_nofilter_labelsCountAcr.copy()
df = df.pivot(index="county", columns="label", values=plot_col).reset_index(drop=False)
df.fillna(0, inplace=True)
df.sort_values(by=["county"], inplace=True)
df.reset_index(drop=True, inplace=True)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, gridspec_kw={"hspace": 0.35, "wspace": 0.05})
axs.grid(axis="y", which="both")
X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5 * bar_width_
X_axis = np.array(range(0, step_size_ * len(df.county), step_size_))

axs.bar(X_axis - bar_width_ * 2, df["double-cropped"], color=color_dict["double-cropped"],
        width=bar_width_, label="double-cropped",)

axs.bar(X_axis - bar_width_, df["single-cropped"], color=color_dict["single-cropped"],
        width=bar_width_, label="single-cropped")

axs.tick_params(axis="x", labelrotation=90)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("acreage")
axs.legend(loc="best")
axs.xaxis.set_ticks_position("none")

# send the guidelines back
ymin, ymax = axs.get_ylim();
axs.set(ylim=(ymin - 1, ymax + 25), axisbelow=True);


# %%
len(X_axis - bar_width_)

# %%
df["single-cropped"].shape

# %%

# %%
tick_legend_FontSize = 10
params = {
    "legend.fontsize": tick_legend_FontSize * 1.5,  # medium, large
    "axes.labelsize": tick_legend_FontSize * 1.7,
    "axes.titlesize": tick_legend_FontSize * 1.7,
    "xtick.labelsize": tick_legend_FontSize * 1.5,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.5,  #  * 0.75
    "axes.titlepad": 10,
    "font.size": 14,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)


# %%
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
plot_col = "acres"
df = county_nofilter_labelsCountAcr.copy()
df = df.pivot(index="county", columns="label", values=plot_col).reset_index(drop=False)
df.fillna(0, inplace=True)
df.sort_values(by=["county"], inplace=True)
df.reset_index(drop=True, inplace=True)
counties = list(df.county.unique())

x = np.arange(len(counties))  # the label locations
width, multiplier = 0.35, 0  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=False,  # sharey='col', # sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.35, "wspace": 0.05})
ax.grid(axis="y", which="both")

for a_col in ["double-cropped", "single-cropped"]:
    offset = width * multiplier
    rects = ax.bar(x + offset, df[a_col], width, label=a_col)
    ax.bar_label(rects, padding=3, label_type="edge")
    multiplier += 1

ax.set_ylim([0, 550000])
ax.set_ylabel(plot_col)
ax.set_xticks(x + width, counties)
ax.legend(loc="upper left", ncols=1)
ax.tick_params(axis="x", labelrotation=90)
file_name = data_dir_ + "county_2022_nofilter_labelsAcr.pdf"
# plt.savefig(fname=file_name, dpi=200, bbox_inches="tight", transparent=False)
plt.show()

# %%

# %%
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
plot_col = "field_count"
df = county_nofilter_labelsCountAcr.copy()
df = df.pivot(index="county", columns="label", values=plot_col).reset_index(drop=False)
df.fillna(0, inplace=True)
df.sort_values(by=["county"], inplace=True)
df.reset_index(drop=True, inplace=True)
counties = list(df.county.unique())

x, multiplier = np.arange(len(counties)), 0  # the label locations

fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=False,  # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={"hspace": 0.35, "wspace": 0.05})
ax.grid(axis="y", which="both")

for a_col in ["double-cropped", "single-cropped"]:
    offset = width * multiplier
    rects = ax.bar(x + offset, df[a_col], width, label=a_col)
    ax.bar_label(rects, padding=3, label_type="edge")
    multiplier += 1

ax.set_ylim([0, 20000])
ax.set_ylabel(plot_col)
ax.set_xticks(x + width, counties)
ax.legend(loc="upper left", ncols=1)
ax.tick_params(axis="x", labelrotation=90)
file_name = data_dir_ + "county_2022_nofilter_labelsCount.pdf"
# plt.savefig(fname=file_name, dpi=200, bbox_inches="tight", transparent=False)
plt.show()

# %%

# %% [markdown]
# # Put all seed crops in one category

# %%
seed_idx = all_correct_year.loc[all_correct_year["croptyp"].str.contains("seed")].index
all_correct_year.loc[seed_idx, "croptyp"] = "seed crops"

# %%
potential_2D = [
    "alfalfa hay",
    "alfalfa/grass hay",
    "barley",
    "barley hay",
    "bean, dry",
    "bean, garbanzo",
    "bean, green",
    "buckwheat",
    "canola",
    "carrot",
    "clover/grass hay",
    "corn, field",
    "corn, sweet",
    "grass hay",
    "hops",
    "market crops",
    "oat",
    "oat hay",
    "onion",
    "pasture",
    "pea hay",
    "pea, dry",
    "pea, green",
    "potato",
    "pumpkin",
    "rye",
    "rye hay",
    "seed crops",
    "soybean",
    "spinach",
    "sudangrass",
    "sunflower",
    "timothy",
    "tomato",
    "triticale",
    "triticale hay",
    "wheat",
    "wheat fallow",
    "wheat hay",
    "yellow mustard",
]

perennials = [
    x
    for x in sorted(list(all_correct_year.croptyp.unique()))
    if not (x in potential_2D)
]

# %%
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
yr = 2022
plot_col = "id"

df = pd.DataFrame(
    all_correct_year[all_correct_year["image_year"] == yr].groupby(["croptyp", "label"])["id"].count()
).reset_index()
df = df[df.croptyp.isin(potential_2D)]
y_lim_max_ = df[plot_col].max() + df[plot_col].max() * 0.12

df = df.pivot(index="croptyp", columns="label", values=plot_col).reset_index(drop=False)
df.fillna(0, inplace=True)
df.sort_values(by=["croptyp"], inplace=True)
df.reset_index(drop=True, inplace=True)
counties = list(df["croptyp"].unique())

x, multiplier = np.arange(len(counties)), 0  # the label locations

fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=False,  # sharey='col', # sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.35, "wspace": 0.05})
ax.grid(axis="y", which="both")

for a_col in ["double-cropped", "single-cropped"]:
    offset = width * multiplier
    rects = ax.bar(x + offset, df[a_col], width, label=a_col)
    ax.bar_label(rects, padding=3, label_type="edge")
    multiplier += 1

ax.set_ylim([0, y_lim_max_])
if plot_col == "id":
    ax.set_ylabel("field count")
else:
    ax.set_ylabel(plot_col)

ax.set_xticks(x + width, counties)
ax.legend(loc="upper left", ncols=1)
ax.tick_params(axis="x", labelrotation=90)
file_name = data_dir_ + "crop_2022_filter_labelsCount_potential2D.pdf"
# plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
plt.show()

# %%
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
plot_col = "id"
df = pd.DataFrame(
    all_correct_year[all_correct_year["image_year"] == yr].groupby(["croptyp", "label"])["id"].count()
).reset_index()
df = df[df.croptyp.isin(perennials)]
y_lim_max_ = df[plot_col].max() + df[plot_col].max() * 0.12

df = df.pivot(index="croptyp", columns="label", values=plot_col).reset_index(drop=False)
df.fillna(0, inplace=True)
df.sort_values(by=["croptyp"], inplace=True)
df.reset_index(drop=True, inplace=True)
counties = list(df["croptyp"].unique())

x, multiplier = np.arange(len(counties)), 0  # the label locations

fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=False,  # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={"hspace": 0.35, "wspace": 0.05})
ax.grid(axis="y", which="both")

for a_col in ["double-cropped", "single-cropped"]:
    offset = width * multiplier
    rects = ax.bar(x + offset, df[a_col], width, label=a_col)
    ax.bar_label(rects, padding=5, label_type="edge")
    multiplier += 1

ax.set_ylim([0, y_lim_max_])
if plot_col == "id":
    ax.set_ylabel("field count")
else:
    ax.set_ylabel(plot_col)

ax.set_xticks(x + width, counties)
ax.legend(loc="best", ncols=1)
ax.tick_params(axis="x", labelrotation=90)
file_name = data_dir_ + "crop_2022_filter_labelsCount_perennials.pdf"
# plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
plt.show()

# %% [markdown]
# ## Filtered tables

# %%
yr = 2021
L = len(all_correct_year[all_correct_year["image_year"] == yr]["id"].unique())
print("total number of fields is [{}].".format(L))
print("total acre is [{}].".format(all_correct_year[all_correct_year["image_year"] == yr]["acres"].sum()))

# %%
pd.DataFrame(all_correct_year[all_correct_year["image_year"] == yr].groupby("label")["id"].count()).reset_index()

# %%
pd.DataFrame(all_correct_year[all_correct_year["image_year"] == yr].groupby("label")["acres"].sum()).reset_index()

# %% [markdown]
# # Write for-loops
#
# There are too may options:
#   - filtered or not
#   - 2022 or 2023
#   - crop-wise or county-wise
#     - in case of crop-wise: 2D or perennials
#    - Acres or fiel-count

# %% [markdown]
# ## For-loop for county-wise
#
# Since we do not have 2D or perennials I am writing two for-loops. Easier to manage the loops as well.

# %%
params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    "axes.labelsize": tick_legend_FontSize * 1,
    "axes.titlesize": tick_legend_FontSize * 1,
    "xtick.labelsize": tick_legend_FontSize,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize,
}
plt.rcParams.update(params)

params = {
    "legend.fontsize": tick_legend_FontSize * 1.5,  # medium, large
    "axes.labelsize": tick_legend_FontSize * 1.7,
    "axes.titlesize": tick_legend_FontSize * 1.7,
    "xtick.labelsize": tick_legend_FontSize * 1.5,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.5,
}
plt.rcParams.update(params)

# %%

# %%
filter_ = [True, False]
plot_what = ["id", "acres"]

counter = 1
for a_year in years:
    for a_filter in filter_:
        for plot_col in plot_what:
            if plot_col == "id":
                if a_filter == True:
                    df = all_correct_year[all_correct_year["image_year"] == a_year].copy()
                elif a_filter == False:                    
                    df = all_no_filter[all_no_filter["image_year"] == a_year].copy()

                y_label_ = "field count"
                df = pd.DataFrame(df.groupby(["county", "label"])["id"].count()).reset_index()
                y_lim_max_ = df[plot_col].max() + df[plot_col].max() * 0.12

                df = df.pivot(index="county", columns="label", values=plot_col).reset_index(drop=False)
                df.fillna(0, inplace=True)
                df.sort_values(by=["county"], inplace=True)
                df.reset_index(drop=True, inplace=True)
                counties = list(df["county"].unique())

                x = np.arange(len(counties))  # the label locations
                width, multiplier = 0.25, 0  # the width of the bars

                fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=False,
                                       gridspec_kw={"hspace": 0.35, "wspace": 0.05})
                ax.grid(axis="y", which="both")

                for a_col in ["double-cropped", "single-cropped"]:
                    offset = width * multiplier
                    rects = ax.bar(x + offset, df[a_col], width, label=a_col)
                    ax.bar_label(rects, padding=3, label_type="edge")
                    multiplier += 1

                ax.set_ylim([0, y_lim_max_])
                if plot_col == "id":
                    ax.set_ylabel(y_label_)
                else:
                    ax.set_ylabel(plot_col)

                ax.set_xticks(x + width, counties)
                ax.legend(loc="best", ncols=1)
                ax.tick_params(axis="x", labelrotation=90)
                ymin, ymax = ax.get_ylim()  # send the guidelines back
                ax.set(ylim=(ymin - 1, ymax + 25), axisbelow=True)
                # send the guidelines back
                ax.set_title(f"Year {a_year}. srvy filter: {a_filter}. {y_label_}")
                file_name = plot_dir+ f"county_{a_year}_filter{a_filter}_{y_label_.replace(' ', '_')}.pdf"

                plt.savefig(fname=file_name, dpi=200, bbox_inches="tight", transparent=False)
                plt.close()

            if plot_col == "acres":
                if a_filter == True:
                    df = all_correct_year[all_correct_year["image_year"] == a_year].copy()
                elif a_filter == False:
                    df = all_no_filter[all_no_filter["image_year"] == a_year].copy()
                    
                df = df[["county", "label", "acres"]]
                df = pd.DataFrame(df.groupby(["county", "label"])["acres"].sum()).reset_index()
                y_lim_max_ = df[plot_col].max() + df[plot_col].max() * 0.12
                df = df.pivot(index="county", columns="label", values=plot_col).reset_index(drop=False)

                df.fillna(0, inplace=True)
                df.sort_values(by=["county"], inplace=True)
                df.reset_index(drop=True, inplace=True)
                counties = list(df.county.unique())

                x = np.arange(len(counties))  # the label locations
                width, multiplier = 0.35, 0  # the width of the bars

                fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=False,
                                       gridspec_kw={"hspace": 0.35, "wspace": 0.05})
                ax.grid(axis="y", which="both")

                for a_col in ["double-cropped", "single-cropped"]:
                    offset = width * multiplier
                    rects = ax.bar(x + offset, df[a_col], width, label=a_col)
                    ax.bar_label(rects, padding=3, label_type="edge")
                    multiplier += 1

                ax.set_ylim([0, y_lim_max_])
                ax.set_ylabel(plot_col)
                ax.set_xticks(x + width, counties)
                ax.legend(loc="best", ncols=1)
                ax.tick_params(axis="x", labelrotation=90)

                ymin, ymax = ax.get_ylim()  # send the guidelines back
                ax.set(ylim=(ymin - 1, ymax + 25), axisbelow=True)
                # send the guidelines back

                ax.set_title(f"Year {a_year}. srvy filter: {a_filter}. {plot_col}")
                file_name = (plot_dir + f"county_{a_year}_filter{a_filter}_{plot_col}.pdf")

                plt.savefig(fname=file_name, dpi=200, bbox_inches="tight", transparent=False)
                plt.close()

# %%

# %% [markdown]
# # Crop-wise plots
#
# filter_ must be True

# %%
params = {
    "legend.fontsize": tick_legend_FontSize * 1.5,  # medium, large
    "axes.labelsize": tick_legend_FontSize * 1.7,
    "axes.titlesize": tick_legend_FontSize * 1.7,
    "xtick.labelsize": tick_legend_FontSize * 1.5,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.5,
}
plt.rcParams.update(params)

# %%
perennials_choice = [True, False]
filter_ = [True]
counter = 1

for a_year in years:
    for a_filter in filter_:
        for plot_col in plot_what:
            for perennial in perennials_choice:
                if plot_col == "id":
                    if a_filter == True:
                        df = all_correct_year[all_correct_year["image_year"] == a_year].copy()
                    elif a_filter == False:
                        df = all_no_filter[all_no_filter["image_year"] == a_year].copy()
                        
                    if perennial == True:
                        df = df[df.croptyp.isin(perennials)].copy()
                        lastName = "perennial"
                        plot_width_ = 20
                    else:
                        df = df[df.croptyp.isin(potential_2D)].copy()
                        lastName = "2D"
                        plot_width_ = 30

                    y_label_ = "acres"
                    df = pd.DataFrame(df.groupby(["croptyp", "label"])["id"].count()).reset_index()
                    y_lim_max_ = df[plot_col].max() + df[plot_col].max() * 0.12

                    df = df.pivot(index="croptyp", columns="label", values=plot_col).reset_index(drop=False)
                    df.fillna(0, inplace=True)
                    df.sort_values(by=["croptyp"], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    counties = list(df["croptyp"].unique())

                    x = np.arange(len(counties))  # the label locations
                    width, multiplier = 0.25, 0  # the width of the bars

                    fig, ax = plt.subplots(1, 1, figsize=(plot_width_, 3), sharex=False,
                        gridspec_kw={"hspace": 0.35, "wspace": 0.05})
                    ax.grid(axis="y", which="both")

                    for a_col in ["double-cropped", "single-cropped"]:
                        offset = width * multiplier
                        rects = ax.bar(x + offset, df[a_col], width, label=a_col)
                        ax.bar_label(rects, padding=3, label_type="edge")
                        multiplier += 1

                    ax.set_ylim([0, y_lim_max_])
                    if plot_col == "id":
                        ax.set_ylabel(y_label_)
                    else:
                        ax.set_ylabel(plot_col)

                    ax.set_xticks(x + width, counties)
                    ax.legend(loc="best", ncols=1)
                    ax.tick_params(axis="x", labelrotation=90)
                    ax.set_title(f"Year {a_year}. srvy filter: {a_filter}. {y_label_}")

                    ymin, ymax = ax.get_ylim()  # send the guidelines back
                    ax.set(ylim=(ymin - 1, ymax + 25), axisbelow=True)
                    # send the guidelines back

                    file_name = plot_dir+\
                            f"crop_{a_year}_filter{a_filter}_{y_label_.replace(' ', '_')}_{lastName}.pdf"

                    plt.savefig(fname=file_name, dpi=200, bbox_inches="tight", transparent=False)
                    plt.close()

                if plot_col == "acres":
                    if a_filter == True:
                        df = all_correct_year[all_correct_year["image_year"] == a_year].copy()
                    elif a_filter == False:
                        df = all_no_filter[all_no_filter["image_year"] == a_year].copy()

                    if perennial == True:
                        df = df[df.croptyp.isin(perennials)].copy()
                        lastName = "perennial"
                    else:
                        df = df[df.croptyp.isin(potential_2D)].copy()
                        lastName = "2D"
                    df = pd.DataFrame(df.groupby(["croptyp", "label"])["acres"].sum()).reset_index()
                    y_lim_max_ = df[plot_col].max() + df[plot_col].max() * 0.12
                    df = df.pivot(index="croptyp", columns="label", values=plot_col).reset_index(drop=False)

                    df.fillna(0, inplace=True)
                    df.sort_values(by=["croptyp"], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    counties = list(df.croptyp.unique())

                    x = np.arange(len(counties))  # the label locations
                    width, multiplier = 0.35, 0  # the width of the bars

                    fig, ax = plt.subplots(1, 1, figsize=(plot_width_, 3), sharex=False,
                        gridspec_kw={"hspace": 0.35, "wspace": 0.05})
                    ax.grid(axis="y", which="both")

                    for a_col in ["double-cropped", "single-cropped"]:
                        offset = width * multiplier
                        rects = ax.bar(x + offset, df[a_col], width, label=a_col)
                        ax.bar_label(rects, padding=3, label_type="edge")
                        multiplier += 1

                    ax.set_ylim([0, y_lim_max_])
                    ax.set_ylabel(plot_col)
                    ax.set_xticks(x + width, counties)
                    ax.legend(loc="best", ncols=1)
                    ax.tick_params(axis="x", labelrotation=90)
                    ymin, ymax = ax.get_ylim()  # send the guidelines back
                    ax.set(ylim=(ymin - 1, ymax + 25), axisbelow=True)
                    # send the guidelines back

                    ax.set_title(f"Year {a_year}. srvy filter: {a_filter}. {y_label_}")
                    file_name = (plot_dir+\
                        f"crop_{a_year}_filter{a_filter}_{plot_col}_{lastName}.pdf")

                    plt.savefig(fname=file_name, dpi=200, bbox_inches="tight", transparent=False)
                    plt.close()

# %%
len(perennials)

# %%
len(potential_2D)

# %%

# %%
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
