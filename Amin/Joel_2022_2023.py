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
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow
import sys, os, os.path,  pickle, time

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
data_dir_ = "/Users/hn/Documents/01_research_data/Amin/Joel/"

# %%
data_2022_nofilter = pd.read_csv(data_dir_ + "data_2022_nofilter.csv")
data_2023_nofilter = pd.read_csv(data_dir_ + "data_2023_nofilter.csv")

data_2022_nofilter['CropTyp'] = data_2022_nofilter['CropTyp'].str.lower()
data_2023_nofilter['CropTyp'] = data_2023_nofilter['CropTyp'].str.lower()

data_2022_nofilter.drop(columns=['Unnamed: 0'], inplace=True)
data_2023_nofilter.drop(columns=['Unnamed: 0'], inplace=True)



### Rename column names: lower case for consistency
data_2022_nofilter.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
data_2023_nofilter.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

data_2023_nofilter.head(2)


##### Sort by id
data_2022_nofilter.sort_values(by=["id"], inplace=True)
data_2023_nofilter.sort_values(by=["id"], inplace=True)

data_2022_nofilter.reset_index(drop=True, inplace=True)
data_2023_nofilter.reset_index(drop=True, inplace=True)

data_2023_nofilter.head(2)

# %%
### Convert type of lstsrvd from string to date

data_2022_nofilter.lstsrvd = pd.to_datetime(data_2022_nofilter.lstsrvd)

data_2023_nofilter.lstsrvd = pd.to_datetime(data_2023_nofilter.lstsrvd)
data_2023_nofilter.head(2)

# %%
data_2022_nofilter["last_survey_year"] = data_2022_nofilter.lstsrvd.dt.year
data_2023_nofilter["last_survey_year"] = data_2023_nofilter.lstsrvd.dt.year
data_2023_nofilter.head(2)

data_2022_nofilter["image_year"] = 2022
data_2023_nofilter["image_year"] = 2023

data_2023_nofilter.head(2)

# %%
len(sorted(data_2022_nofilter.croptyp.unique()))

# %%
all_crops = sorted(data_2023_nofilter.croptyp.unique())

# %% [markdown]
# # Export data for Jeol and Kirti

# %%
data_2022_2023 = pd.concat([data_2023_nofilter, data_2022_nofilter])
data_2022_2023.shape

# %%
data_2022_2023["correct_year"] = "False"
data_2022_2023.loc[data_2022_2023.last_survey_year == data_2022_2023.image_year, "correct_year"] = "True"
data_2022_2023.head(2)

# %%
data_2022_2023.to_csv(data_dir_ + "double_cropLabels_2022_2023.csv", index=False)

# %%

# %%
# Check w/ Kirti.
# Some of these are new to me
# Is tea and kiwi produced here?!

bad_crops_2Drop = ['0',
                   'CRP/Conservation',
                   'Christmas Tree',
                   'Cover Crop',
                   'Dandelion',
                   'Developed',
                   'Driving Range',
                   'Echinacea',
#                   'Fallow',
#                   'Fallow, Idle',
#                   'Fallow, Tilled',
                   'Golf Course',
                   'Hemp',
                   'Herb, Unknown',
                   'Kiwi',
#                   'Nursery, Caneberry',
#                   'Nursery, Greenhouse',
#                   'Nursery, Lavender',
#                   'Nursery, Orchard/Vineyard',
#                   'Nursery, Ornamental',
                   'Peony',
                   'Reclamation Seed',
                   'Research Station',
                   'Silviculture',
                   'Sod Farm',
                   'Tea',
                   'Unknown',
                   'Wildlife Feed']
bad_crops_2Drop = [x.lower() for x in bad_crops_2Drop]

# %%
data_2022_nofilter = data_2022_nofilter[~data_2022_nofilter.croptyp.isin(bad_crops_2Drop)]
data_2023_nofilter = data_2023_nofilter[~data_2023_nofilter.croptyp.isin(bad_crops_2Drop)]

data_2022_nofilter.reset_index(drop=True, inplace=True)
data_2023_nofilter.reset_index(drop=True, inplace=True)

# %%
print (f"{data_2022_nofilter.shape = }")
print (f"{data_2023_nofilter.shape = }")
print ()

print (f"{len(data_2022_nofilter.id.unique()) = }")
print (f"{len(data_2023_nofilter.id.unique()) = }")

# %%
data_2022_surveyFilter = data_2022_nofilter[data_2022_nofilter.last_survey_year == \
                                            data_2022_nofilter.image_year].copy()


data_2023_surveyFilter = data_2023_nofilter[data_2023_nofilter.last_survey_year == \
                                            data_2023_nofilter.image_year].copy()

data_2023_surveyFilter.head(2)

# %%
print (f"{data_2022_nofilter.shape = }")
print (f"{data_2022_surveyFilter.shape = }")
print ()
print (f"{data_2023_nofilter.shape = }")
print (f"{data_2023_surveyFilter.shape = }")

# %% [markdown]
# # Create (at least) 4 tables
#
# Two tables for each year
#
# - One table say how many/acres are double cropped in general.
# - One table say how many/acres are double cropped using proper survey date so that they can dive in and see things based on crop-type and whatnot!!!
# - Extra tables can have crop types in it as well.

# %% [markdown]
# ### No-Filter tables 2022

# %%
print ("total acre is [{}].".format(data_2022_nofilter["acres"].sum()))

# %%
pd.DataFrame(data_2022_nofilter.groupby("label")["id"].count()).reset_index()

# %%
pd.DataFrame(data_2022_nofilter.groupby("label")["acres"].sum()).reset_index()

# %% [markdown]
# ### No-Filter tables 2022: counties

# %%
field_count_counties = pd.DataFrame(data_2022_nofilter.groupby(["county"])["id"].count()).reset_index()
field_count_counties.rename(columns={"id":"total_field_count"}, inplace=True)

field_acr_counties = pd.DataFrame(data_2022_nofilter.groupby(["county"])["acres"].sum()).reset_index()
field_acr_counties.rename(columns={"id":"acres"}, inplace=True)

county_field_countAcr = pd.merge(field_count_counties, field_acr_counties, on=['county'], how='left')
county_field_countAcr.head(2)

# %%

# %%
data_2022_nofilter_labelCounts = pd.DataFrame(data_2022_nofilter.groupby(["county", "label"])\
                                              ["id"].count()).reset_index()
data_2022_nofilter_labelCounts.rename(columns={"id":"field_count"}, inplace=True)
data_2022_nofilter_labelCounts.head(2)

data_2022_nofilter_labelAcr = pd.DataFrame(data_2022_nofilter.groupby(["county", "label"])\
                                              ["acres"].sum()).reset_index()
data_2022_nofilter_labelAcr.head(2)

county_2022_nofilter_labelsCountAcr = pd.merge(data_2022_nofilter_labelAcr, data_2022_nofilter_labelCounts, 
                                            on=['county', "label"], how='left')
county_2022_nofilter_labelsCountAcr.head(2)

# %%
data_2023_nofilter_labelCounts = pd.DataFrame(data_2023_nofilter.groupby(["county", "label"])\
                                              ["id"].count()).reset_index()
data_2023_nofilter_labelCounts.rename(columns={"id":"field_count"}, inplace=True)
data_2023_nofilter_labelCounts.head(2)

data_2023_nofilter_labelAcr = pd.DataFrame(data_2023_nofilter.groupby(["county", "label"])\
                                              ["acres"].sum()).reset_index()
data_2023_nofilter_labelAcr.head(2)

county_2023_nofilter_labelsCountAcr = pd.merge(data_2023_nofilter_labelAcr, data_2023_nofilter_labelCounts, 
                                            on=['county', "label"], how='left')
county_2023_nofilter_labelsCountAcr.head(2)

# %%
tick_legend_FontSize = 10

params = {'legend.fontsize': tick_legend_FontSize, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': tick_legend_FontSize*1.2,
          'axes.titlesize': tick_legend_FontSize*1.3,
          'xtick.labelsize': tick_legend_FontSize, #  * 0.75
          'ytick.labelsize': tick_legend_FontSize, #  * 0.75
          'axes.titlepad': 10}

plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)

color_dict = {"single-cropped": "#DDCC77",
              "double-cropped": "#332288",
             }

color_dict = {"single-cropped": "dodgerblue",
              "double-cropped": "red",
             }

# %%
plot_col = "acres"
df = county_2022_nofilter_labelsCountAcr.copy()
df = df.pivot(index='county', columns='label', values=plot_col).reset_index(drop=False)
df.columns = df.columns.values
df.plot(x='county', kind='bar', stacked=False); # title=plot_col
plt.xlabel('county');
plt.ylabel(plot_col);

# %%

# %%
import plotly.express as px
df = county_2022_nofilter_labelsCountAcr.copy()
fig = px.bar(df, x="county", y="acres",
             color='label', barmode='group', text="acres", height=400)

fig.update_xaxes(categoryorder='array', 
                 categoryarray= df.county.unique())

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
df = county_2022_nofilter_labelsCountAcr.copy()
df = df.pivot(index='county', columns='label', values=plot_col).reset_index(drop=False)
df.fillna(0, inplace=True)
df.sort_values(by=["county"], inplace=True)
df.reset_index(drop=True, inplace=True)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df["double-cropped"], 
        color = color_dict["double-cropped"], width = bar_width_, label="double-cropped")

axs.bar(X_axis - bar_width_, df["single-cropped"], 
        color = color_dict["single-cropped"], width = bar_width_, label="single-cropped")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.county);


axs.set_ylabel("acreage")
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+25), axisbelow=True); # send the guidelines back

# %%
tick_legend_FontSize = 10
params = {'legend.fontsize': tick_legend_FontSize*1.5, # medium, large
          'axes.labelsize': tick_legend_FontSize*1.7,
          'axes.titlesize': tick_legend_FontSize*1.7,
          'xtick.labelsize': tick_legend_FontSize*1.5, #  * 0.75
          'ytick.labelsize': tick_legend_FontSize*1.5, #  * 0.75
          'axes.titlepad': 10,
          'font.size' : 14}

plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)


# %%

# %% [markdown]
# ## No-Filter tables 2023

# %%
print ("total acre is [{}].".format(data_2022_nofilter["acres"].sum()))

# %%
pd.DataFrame(data_2023_nofilter.groupby("label")["id"].count()).\
reset_index().rename(columns={"id":"total_field_count"})

# %%
pd.DataFrame(data_2023_nofilter.groupby("label")["acres"].sum()).reset_index()

# %%

# %% [markdown]
# ## Filtered 2022

# %%
L = len(data_2022_surveyFilter["id"].unique())
print ("total number of fields is [{}].".format(L))
print ("total acre is [{}].".format(data_2022_surveyFilter["acres"].sum()))

# %%
pd.DataFrame(data_2022_surveyFilter.groupby("label")["id"].count()).reset_index()

# %%
pd.DataFrame(data_2022_surveyFilter.groupby("label")["acres"].sum()).reset_index()

# %% [markdown]
# ## Crop Specific 2022

# %%
pd.DataFrame(data_2022_surveyFilter.groupby(["croptyp", "label"])["id"].count()).reset_index()\
.rename(columns={"id":"field_count"})

# %%
pd.DataFrame(data_2022_surveyFilter.groupby(["croptyp", "label"])["acres"].sum()).reset_index()

# %% [markdown]
# # Put all seed crops in one category

# %%
seed_idx = data_2022_surveyFilter.loc[data_2022_surveyFilter['croptyp'].str.contains("seed")].index
data_2022_surveyFilter.loc[seed_idx, "croptyp"] = "seed crops"


seed_idx = data_2023_surveyFilter.loc[data_2023_surveyFilter['croptyp'].str.contains("seed")].index
data_2023_surveyFilter.loc[seed_idx, "croptyp"] = "seed crops"

# %%
potential_2D = ['alfalfa hay',
                'alfalfa/grass hay',
                'barley',
                'barley hay',
                'bean, dry',
                'bean, garbanzo',
                'bean, green',
                'buckwheat',
                'canola',
                'carrot',
                'clover/grass hay',
                'corn, field',
                'corn, sweet',
                'grass hay',
                'hops',
                'market crops',
                'oat',
                'oat hay',
                'onion',
                'pasture',
                'pea hay',
                'pea, dry',
                'pea, green',
                'potato',
                'pumpkin',
                'rye',
                'rye hay',
                'seed crops',
                'soybean',
                'spinach',
                'sudangrass',
                'sunflower',
                'timothy',
                'tomato',
                'triticale',
                'triticale hay',
                'wheat',
                'wheat fallow',
                'wheat hay',
                'yellow mustard']

perennials = [x for x in sorted(list(data_2022_surveyFilter.croptyp.unique())) if not(x in potential_2D)]

# %%

# %%

# %% [markdown]
# ## Filtered tables 2023

# %%
L = len(data_2023_surveyFilter["id"].unique())
print ("total number of fields is [{}].".format(L))
print ("total acre is [{}].".format(data_2023_surveyFilter["acres"].sum()))

# %%
pd.DataFrame(data_2023_surveyFilter.groupby("label")["id"].count()).reset_index()

# %%
pd.DataFrame(data_2023_surveyFilter.groupby("label")["acres"].sum()).reset_index()

# %% [markdown]
# # Crop Specific 2023

# %%

# %%

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
params = {'legend.fontsize': tick_legend_FontSize, # medium, large
          'axes.labelsize': tick_legend_FontSize*1,
          'axes.titlesize': tick_legend_FontSize*1,
          'xtick.labelsize': tick_legend_FontSize, #  * 0.75
          'ytick.labelsize': tick_legend_FontSize}
plt.rcParams.update(params)

params = {'legend.fontsize': tick_legend_FontSize*1.5, # medium, large
          'axes.labelsize': tick_legend_FontSize*1.7,
          'axes.titlesize': tick_legend_FontSize*1.7,
          'xtick.labelsize': tick_legend_FontSize*1.5, #  * 0.75
          'ytick.labelsize': tick_legend_FontSize*1.5}
plt.rcParams.update(params)

# %%
filter_ = [True, False]
years = [2022, 2023]
plot_what = ["id", "acres"]

counter = 1
for a_year in years:
    for a_filter in filter_:
        for plot_col in plot_what:
            if plot_col == "id":
                if a_year == 2022 and a_filter==True:
                    df = data_2022_surveyFilter.copy()
                elif a_year == 2023 and a_filter==True:
                    df = data_2023_surveyFilter.copy()
                elif a_year == 2022 and a_filter==False:
                    df = data_2022_nofilter.copy()
                elif a_year == 2023 and a_filter==False:
                    df = data_2023_nofilter.copy()
            
                y_label_ = "field count"
                df = pd.DataFrame(df.groupby(["county", "label"])["id"].count()).reset_index()
                y_lim_max_ = df[plot_col].max() + df[plot_col].max() * .12

                df = df.pivot(index='county', columns='label', values=plot_col).reset_index(drop=False)
                df.fillna(0, inplace=True)
                df.sort_values(by=["county"], inplace=True)
                df.reset_index(drop=True, inplace=True)
                counties = list(df["county"].unique())

                x = np.arange(len(counties)) # the label locations
                width = 0.25  # the width of the bars
                multiplier = 0

                fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=False,\
                                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
                ax.grid(axis='y', which='both')

                for a_col in ["double-cropped", "single-cropped"]:
                    offset = width * multiplier
                    rects = ax.bar(x + offset, df[a_col], width, label=a_col)
                    ax.bar_label(rects, padding=3, label_type='edge')
                    multiplier += 1

                ax.set_ylim([0, y_lim_max_])
                if plot_col=="id":
                    ax.set_ylabel(y_label_)
                else:
                    ax.set_ylabel(plot_col)

                ax.set_xticks(x + width, counties)
                ax.legend(loc='best', ncols=1)
                ax.tick_params(axis='x', labelrotation = 90)
                ymin, ymax = ax.get_ylim() # send the guidelines back
                ax.set(ylim=(ymin-1, ymax+25), axisbelow=True); # send the guidelines back
                ax.set_title(f'Year {a_year}. srvy filter: {a_filter}. {y_label_}')
                file_name = data_dir_ + "plots/" + \
                               f"county_{a_year}_filter{a_filter}_{y_label_.replace(' ', '_')}.pdf"

                plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
                plt.close()
                
            if plot_col == "acres":
                if a_year == 2022 and a_filter==True:
                    df = data_2022_surveyFilter.copy()
                elif a_year == 2023 and a_filter==True:
                    df = data_2023_surveyFilter.copy()
                elif a_year == 2022 and a_filter==False:
                    df = data_2022_nofilter.copy()
                elif a_year == 2023 and a_filter==False:
                    df = data_2023_nofilter.copy()
                    
                df = df[["county", "label", "acres"]]
                df = pd.DataFrame(df.groupby(["county", "label"])["acres"].sum()).reset_index()
                y_lim_max_ = df[plot_col].max() + df[plot_col].max() * .12
                df = df.pivot(index='county', columns='label', values=plot_col).reset_index(drop=False)

                df.fillna(0, inplace=True)
                df.sort_values(by=["county"], inplace=True)
                df.reset_index(drop=True, inplace=True)
                counties = list(df.county.unique())

                x = np.arange(len(counties)) # the label locations
                width = 0.35  # the width of the bars
                multiplier = 0

                fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=False, \
                                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
                ax.grid(axis='y', which='both')

                for a_col in ["double-cropped", "single-cropped"]:
                    offset = width * multiplier
                    rects = ax.bar(x + offset, df[a_col], width, label=a_col)
                    ax.bar_label(rects, padding=3, label_type='edge')
                    multiplier += 1

                ax.set_ylim([0, y_lim_max_])
                ax.set_ylabel(plot_col)
                ax.set_xticks(x + width, counties)
                ax.legend(loc='best', ncols=1)
                ax.tick_params(axis='x', labelrotation = 90)
                
                ymin, ymax = ax.get_ylim() # send the guidelines back
                ax.set(ylim=(ymin-1, ymax+25), axisbelow=True); # send the guidelines back

                ax.set_title(f'Year {a_year}. srvy filter: {a_filter}. {plot_col}')
                file_name = data_dir_ + "plots/" + \
                               f"county_{a_year}_filter{a_filter}_{plot_col}.pdf"

                plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
                plt.close()

# %% [markdown]
# # Crop-wise plots
#
# filter_ must be True

# %%
params = {'legend.fontsize': tick_legend_FontSize*1.5, # medium, large
          'axes.labelsize': tick_legend_FontSize*1.7,
          'axes.titlesize': tick_legend_FontSize*1.7,
          'xtick.labelsize': tick_legend_FontSize*1.5, #  * 0.75
          'ytick.labelsize': tick_legend_FontSize*1.5}
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
                    if a_year == 2022 and a_filter==True:
                        df = data_2022_surveyFilter.copy()
                    elif a_year == 2023 and a_filter==True:
                        df = data_2023_surveyFilter.copy()
                    elif a_year == 2022 and a_filter==False:
                        df = data_2022_nofilter.copy()
                    elif a_year == 2023 and a_filter==False:
                        df = data_2023_nofilter.copy()
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
                    y_lim_max_ = df[plot_col].max() + df[plot_col].max() * .12

                    df = df.pivot(index='croptyp', columns='label', values=plot_col).reset_index(drop=False)
                    df.fillna(0, inplace=True)
                    df.sort_values(by=["croptyp"], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    counties = list(df["croptyp"].unique())

                    x = np.arange(len(counties)) # the label locations
                    width = 0.25  # the width of the bars
                    multiplier = 0

                    fig, ax = plt.subplots(1, 1, figsize=(plot_width_, 3), sharex=False,\
                                           gridspec_kw={'hspace': 0.35, 'wspace': .05});
                    ax.grid(axis='y', which='both')

                    for a_col in ["double-cropped", "single-cropped"]:
                        offset = width * multiplier
                        rects = ax.bar(x + offset, df[a_col], width, label=a_col)
                        ax.bar_label(rects, padding=3, label_type='edge')
                        multiplier += 1

                    ax.set_ylim([0, y_lim_max_])
                    if plot_col=="id":
                        ax.set_ylabel(y_label_)
                    else:
                        ax.set_ylabel(plot_col)

                    ax.set_xticks(x + width, counties)
                    ax.legend(loc='best', ncols=1)
                    ax.tick_params(axis='x', labelrotation = 90)
                    ax.set_title(f'Year {a_year}. srvy filter: {a_filter}. {y_label_}')
                    
                    ymin, ymax = ax.get_ylim() # send the guidelines back
                    ax.set(ylim=(ymin-1, ymax+25), axisbelow=True); # send the guidelines back

                    file_name = data_dir_ + "plots/" + \
                            f"crop_{a_year}_filter{a_filter}_{y_label_.replace(' ', '_')}_{lastName}.pdf"

                    plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
                    plt.close()

                if plot_col == "acres":
                    if a_year == 2022 and a_filter==True:
                        df = data_2022_surveyFilter.copy()
                    elif a_year == 2023 and a_filter==True:
                        df = data_2023_surveyFilter.copy()
                    elif a_year == 2022 and a_filter==False:
                        df = data_2022_nofilter.copy()
                    elif a_year == 2023 and a_filter==False:
                        df = data_2023_nofilter.copy()
                    
                    if perennial == True:
                        df = df[df.croptyp.isin(perennials)].copy()
                        lastName = "perennial"
                    else:
                        df = df[df.croptyp.isin(potential_2D)].copy()
                        lastName = "2D"
                    df = pd.DataFrame(df.groupby(["croptyp", "label"])["acres"].sum()).reset_index()
                    y_lim_max_ = df[plot_col].max() + df[plot_col].max() * .12
                    df = df.pivot(index='croptyp', columns='label', values=plot_col).reset_index(drop=False)

                    df.fillna(0, inplace=True)
                    df.sort_values(by=["croptyp"], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    counties = list(df.croptyp.unique())

                    x = np.arange(len(counties)) # the label locations
                    width = 0.35  # the width of the bars
                    multiplier = 0

                    fig, ax = plt.subplots(1, 1, figsize=(plot_width_, 3), sharex=False, \
                                           gridspec_kw={'hspace': 0.35, 'wspace': .05});
                    ax.grid(axis='y', which='both')

                    for a_col in ["double-cropped", "single-cropped"]:
                        offset = width * multiplier
                        rects = ax.bar(x + offset, df[a_col], width, label=a_col)
                        ax.bar_label(rects, padding=3, label_type='edge')
                        multiplier += 1

                    ax.set_ylim([0, y_lim_max_])
                    ax.set_ylabel(plot_col)
                    ax.set_xticks(x + width, counties)
                    ax.legend(loc='best', ncols=1)
                    ax.tick_params(axis='x', labelrotation = 90)
                    ymin, ymax = ax.get_ylim() # send the guidelines back
                    ax.set(ylim=(ymin-1, ymax+25), axisbelow=True); # send the guidelines back

                    ax.set_title(f'Year {a_year}. srvy filter: {a_filter}. {y_label_}')
                    file_name = data_dir_ + "plots/" + \
                                   f"crop_{a_year}_filter{a_filter}_{plot_col}_{lastName}.pdf"

                    plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
                    plt.close()

# %%
len(perennials)

# %%
len(potential_2D)

# %%

# %%
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
