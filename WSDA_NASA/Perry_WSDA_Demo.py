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

# %% [markdown] id="KcvoMHLzD1hd"
# #**There is no free lunch**
#
# [Colab resources are not guaranteed](https://research.google.com/colaboratory/faq.html#:~:text=Colab\%20is\%20able\%20to\%20provide,other\%20factors\%20vary\%20over\%20time.). One can, however, [subscribe and increase resources]((https://colab.research.google.com/signup)) at his disposal.
#
# Creating an App also needs $ and human time to create it. Please look at [Quotas and limits](https://cloud.google.com/appengine/docs/standard#:~:text=The%20standard%20environment%20gives%20you,suit%20your%20needs%2C%20see%20Quotas.) section as a starting point!
#
# I ran different counties as a separate jobs and it took hours. If they want to run all counties at once, the cost (computation or $) is even higher.
#
# <p>&nbsp;</p>

# %% colab={"base_uri": "https://localhost:8080/"} id="EjdLZgbOxwlU" outputId="3f7fe70d-8370-4f0d-8f21-3e8048679ac6"
# %who

# %% colab={"base_uri": "https://localhost:8080/"} id="rNWduYF8ELxV" outputId="42b9f09f-0ebb-4f41-8387-c0b2b57cc29b"
# !pip install shutup
import shutup
shutup.please() # kill some of the messages

# %% [markdown] id="1k8_CIo5DqvC"
# # Print Local Time 
#
# colab runs on cloud. So, the time is not our local time.
# This page is useful to determine how to do this.

# %% colab={"base_uri": "https://localhost:8080/"} id="zxAcDFMxDwIm" outputId="0ddb6b56-45be-4d36-f8d7-a4c54bd58859"
# !rm /etc/localtime
# !ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime
# !date

# %% [markdown] id="Nyg94VHsEK8B"
# # **geopandas and geemap must be installed every time!!!**
#

# %% id="F9b71mZbEZQj"
# # !pip install geopandas geemap
# Installs geemap package
import subprocess

try:
    import geemap
except ImportError:
    print('geemap not installed. Must be installed every tim to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

    print('geopandas not installed. Must be installed every time to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geopandas'])
    subprocess.check_call(["python", '-m', 'pip', 'install', 'google.colab'])

# %% [markdown] id="StQCEezUEajV"
# # **Authenticate and import libraries**
#
# We have to impor tthe libraries we need. Moreover, we need to Authenticate every single time!

# %% id="vWONKBNXEpCD"
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
import json, geemap, ee


import scipy # we need this for savitzky-golay

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time, datetime
from datetime import date


try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# %% [markdown] id="Magk2Zr_ExC1"
# ### **Mount Google Drive and import my Python modules**
#
# Here we are importing the Python functions that are written by me and are needed; ```NASA core``` and ```NASA plot core```.
#
# **Note:** These are on Google Drive now. Perhaps we can import them from GitHub.

# %% colab={"base_uri": "https://localhost:8080/"} id="IxshCK3dE8lf" outputId="20400225-eec1-49f9-e9eb-cbccbfc37441"
# Mount YOUR google drive in Colab
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0,"/content/drive/My Drive/Colab Notebooks/")
import NASA_core as nc
import NASA_plot_core as ncp
import GEE_Python_core as gpc

# %% [markdown] id="72Hu_ab3E93Z"
# **Change Current directory to the Colab folder on Google Drive**

# %% id="UEA2gsPJFJ9X"
import os
os.chdir("/content/drive/My Drive/Colab Notebooks/") # Colab Notebooks
# # !ls

# %% [markdown] id="FIrLJ4-pFQEz"
# # Please tell me where to look for the shapefile!

# %% id="juqgx2EpFUU1"
# shp_path = "/Users/hn/Documents/01_research_data/NASA/shapefiles/Grant2017/Grant2017.shp"
# shp_path = "/content/My Drive/NASA_trends/shapefiles/Grant2017/Grant2017.shp"
shp_path = "/content/drive/MyDrive/NASA_trends/shapefiles/Grant2017/Grant2017.shp"
shp_path = "/content/drive/MyDrive/NASA_trends/shapefiles/" + \
            "Grant_4Fields_poly_wCentroids/Grant_4Fields_poly_wCentroids.shp"

# we read our shapefile in to a geopandas data frame using the geopandas.read_file method
# we'll make sure it's initiated in the EPSG 4326 CRS
Grant_4Fields_poly_wCentroids = gpd.read_file(shp_path, crs='EPSG:4326')

# define a helper function to put the geodataframe in the right format for constructing an ee object
# The following function and the immediate line after that works for 1 geometry. not all the fields in the shapefile.
# def shp_to_ee_fmt(geodf):
#     data = json.loads(geodf.to_json())
#     return data['features'][0]['geometry']['coordinates']
# Grant_4Fields_poly_wCentroids = ee.Geometry.MultiPolygon(shp_to_ee_fmt(Grant_4Fields_poly_wCentroids))

# Grant_4Fields_poly_wCentroids = ee.FeatureCollection(Grant_4Fields_poly_wCentroids)

# %% id="C__vgtwHFcBW"
unwanted_columns = ['cntrd_ln', 'cntrd_lt', 'CropGrp', 'Shap_Ar', 'Shp_Lng', 'ExctAcr', 
                    'RtCrpTy', 'TRS', 'Notes', 'IntlSrD']
Grant_4Fields_poly_wCentroids = Grant_4Fields_poly_wCentroids.drop(columns=unwanted_columns)

# %% colab={"base_uri": "https://localhost:8080/", "height": 183} id="EzT2oOS5Fi61" outputId="5fafd5f9-dd65-4854-8c8e-2babf66d1454"
long_eq = "=============================================================================="
print (type(Grant_4Fields_poly_wCentroids))
print (long_eq)
print (f"{Grant_4Fields_poly_wCentroids.shape=}", )
print (long_eq)
Grant_4Fields_poly_wCentroids.head(2)

# %% [markdown] id="JVOsJFXkFoCH"
# # **Form Geographical Regions**
#
#   - First, define a big region that covers Eastern Washington.
#   - Convert shapefile to ```ee.featurecollection.FeatureCollection```.
#

# %% id="Ihy6wVxAHe_A"
xmin = -125.0;
ymin = 45.0;
xmax = -116.0;
ymax = 49.0;

xmed = (xmin + xmax) / 2.0;
ymed = (ymin+ymax) / 2.0;

WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed, ymax], [xmed, ymin], [xmin, ymin]]);
WA2 = ee.Geometry.Polygon([[xmed, ymin], [xmed, ymax], [xmax, ymax], [xmax, ymin], [xmed, ymin]]);
WA = [WA1,WA2];
big_rectangle = ee.FeatureCollection(WA);
SF = geemap.geopandas_to_ee(Grant_4Fields_poly_wCentroids)

# %% [markdown] id="MLucJYy7Hgrr"
# ## **WARNING!**
# For some reason the function ```feature2ee(.)``` does not work ***when*** it is imported from ```core``` module. (However, it works when it is directly written here!!!) So, What the happens with the rest of functions, e.g. smoothing functions, we want to use here?

# %% id="S3GBvrufIuZ8"
# was named "banke" in https://bikeshbade.com.np/tutorials/Detail/?title=Geo-pandas%20data%20frame%20to%20GEE%20feature%20collection%20using%20Python&code=13
# Grant_4Fields_poly_wCentroids_EEFC_from_Func = feature2ee(shp_path)
# Grant_4Fields_poly_wCentroids_EEFC_from_Func = gpc.feature2ee(shp_path)

# %% [markdown] id="_WohFsaWI3_w"
# # Visualize the big region encompassing the Eastern Washington

# %% colab={"base_uri": "https://localhost:8080/", "height": 621, "referenced_widgets": ["b07e72a611ce47088297ab162d91f057", "dab8c76ea9f940a8a27488e8b4867d50", "066ebda074de413d82136f1c53bb99d7", "e148451791ad4fb1ae7fae0fdd03b619", "d40f179d023a4c36ba823429bea91f21", "f85388867ed94447a6b147b0630850a3", "03962ff21ea347bcbd7dc6037e4ac2fa", "2b19544ca31142c3bbf6e55fb2813078", "da262501686148c9a46c67af3ade3bcd", "93a13e97bbc0474f9a39f7f856c838cb", "c48d4930c6354549a85b21a34910ff11", "dd0800d83c444543a41133af963f4b66", "ded1a7fce1eb4bc68862853a28c01050", "93c6fa08019a4f198714bd02316ad6d9", "7a6f714baebf40a289da9d609fcb68db", "3745dfde85a247c08d8888a86ee1e172", "437528841a794e3080a771b5e2ed64b2", "72db5755c51a4d67a8c4664889156ce0", "9b389d23039b4586b65f2994d3960b30", "a6315b5ccd7d4a679132e0ea36698ceb", "d631a36b44f14c0eac374b7b929e98a1", "666fd85ebe4e4070b82195facaeb664d", "7f008939945b4799a7d1b7ca03127ee2", "a270b2cfc6b74e0a8314c2116145d503", "2a97872b80d94e319d1b2c2e3e3709f6", "78aee9476745470d89c880c0255b06c6", "aea068f1a0f94ce8970866ef6cc86bd5", "55993f162a22421b94decd6c7f4c3a38"]} id="n_VdDpeqI66Q" outputId="621116ed-990a-4499-cc05-3baeaec2119f"
Map = geemap.Map(center=[ymed, xmed], zoom=7)
Map.addLayer(WA1, {'color': 'red'}, 'Western Half')
Map.addLayer(WA2, {'color': 'red'}, 'Eastern Half')
Map.addLayer(SF, {'color': 'blue'}, 'Fields')
Map

# %% [markdown] id="OuKWgSEXJD7B"
# # Define Parameters

# %% id="Og1nGEviJJ1_"
start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD
cloud_perc = 70

# %% [markdown] id="rqJXjNo1JWmR"
# ## Collect the needed data from GEE

# %% colab={"base_uri": "https://localhost:8080/"} id="4fOcXB4cJPcC" outputId="772d54ce-36db-412e-ffab-efd6870ea5d4"
imageC = gpc.extract_sentinel_IC(big_rectangle, start_date, end_date, cloud_perc);
print ("The size of [imageC.size().getInfo()] is [{:.0f}].".format(imageC.size().getInfo()))
reduced = gpc.mosaic_and_reduce_IC_mean(imageC, SF, start_date, end_date)

# %% [markdown] id="3g13uBmAJaE1"
# # Export output to Google Driveonly 
#
# We advise you to do it. If Python/CoLab kernel dies, then,
# previous steps should not be repeated.
#
# To save (exporting) time, and space (on Google Drive) you can exclude the following columns for exporting in the cell below and use ```ID``` later to retrieve them if needed.
#
# *   ```Acres```
# *   ```county```
# *   ```CropTyp```
# *   ```DataSrc```
# *   ```Irrigtn```
# *   ```LstSrvD```

# %% colab={"base_uri": "https://localhost:8080/"} id="JktTEw_-JfK4" outputId="ff41c672-6ee1-4ef2-cb52-ca2448b2411f"
# %%time
export_raw_data = True

if export_raw_data==True:
    outfile_name = "Grant_4Fields_poly_wCentroids_colab_output"
    task = ee.batch.Export.table.toDrive(**{
                                        'collection': reduced,
                                        'description': outfile_name,
                                        'folder': "colab_outputs",
                                        'selectors':["ID", "Acres", "county", "CropTyp", "DataSrc", \
                                                     "Irrigtn", "LstSrvD", "EVI", 'NDVI', "system_start_time"],
                                        'fileFormat': 'CSV'})
    task.start()

    import time 
    while task.active():
        print('Polling for task (id: {}). Still breathing'.format(task.id))
        time.sleep(59)

# %% [markdown] id="JWEL1ff9JzI-"
# # **Smooth the data**
#
# This is the end of Earh Engine Part. Below we start smoothing the data and carry on!
#
# First, all these steps can be done behind the scene. But doing them here, one at a time, has the advantage that if something goes wrong in the middle, then
# we do not lose the good stuff that was done earlier!
# For example, of one of the Python libraries/packages needs to be updated in the middle of the way
# we do not have to start doing everything from the beginning!
# <p>&nbsp;</p>
#
# Start with converting the type of ```reduced``` from ```ee.FeatureCollection``` to ```dataframe```.
#
# - For some reason when converting the ```ee.FeatureCollection``` to ```dataframe``` the function has a problem with the ```Notes``` column! So, I remove the unnecessary columns.

# %% colab={"base_uri": "https://localhost:8080/", "height": 147} id="ecZ1AcPDKYHP" outputId="220d2eb2-75f2-4a1a-b24d-4d1c08740071"
# See how long it takes to convert a FeatureCollection to dataframe!
# %%time
needed_columns = ["ID", "Acres", "county", "CropTyp", "DataSrc", \
                  "Irrigtn", "LstSrvD", "EVI", 'NDVI', "system_start_time"]

## Saving as little as possible will help with memory need during run time, etc.
needed_columns = ["ID", "EVI", 'NDVI', "system_start_time"]

reduced = geemap.ee_to_pandas(reduced, selectors=needed_columns)

reduced = reduced[needed_columns]
reduced.head(2)

# %% [markdown] id="qUzIpO0SQNv7"
# # Isolate the ```data``` part of the ```shapefile``` for future use.

# %% colab={"base_uri": "https://localhost:8080/"} id="5U-F_asRR_gA" outputId="2f41751a-855c-486e-f57b-0bc2e37c2836"
SF_data = reduced[["ID", "Acres", "county", "CropTyp", \
                   "DataSrc", "Irrigtn", "LstSrvD"]].copy()
SF_data.drop_duplicates(inplace=True)
SF_data.shape

# %% [markdown] id="0-r6CjbMSBx-"
# **NA removal**
#
# Even though logically and intuitively all the bands should be either available or ```NA```, I have seen before that sometimes ```EVI``` is NA while ```NDVI``` is not. Therefore, I had to choose which VI we want to use so that we can clean the data properly. However, I did not see that here.  when I was testing this code for 4 fields.
#
# Another suprising observation was that the output of Colab had more data compared to its JS counterpart!!!

# %% id="sSDOXhv0SKNb"
# reduced = reduced[reduced["system_start_time"].notna()]
reduced = reduced[reduced["EVI"].notna()]
reduced.reset_index(drop=True, inplace=True)

# %% [markdown] id="Z8GEg8bVSKqn"
# Add human readable time to the dataframe

# %% colab={"base_uri": "https://localhost:8080/"} id="TrlVRbS8SPe-" outputId="6ba872a6-c557-476f-d696-268156ee325f"
reduced = nc.add_human_start_time_by_system_start_time(reduced)
reduced.head(2)
reduced.loc[0, "system_start_time"]

# %% [markdown] id="v6ohoPCBSZVL"
# Make a plot for fun.

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="sqXm2ZKkSedK" outputId="8f24ad3f-5857-49c4-d56b-9d7a36100b07"
#  Pick a field
a_field = reduced[reduced.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(axis='y', which="both")

ax.scatter(a_field['human_system_start_time'], a_field["EVI"], s=40, c='#d62728');
ax.plot(a_field['human_system_start_time'], a_field["EVI"], 
        linestyle='-',  linewidth=3.5, color="#d62728", alpha=0.8,
        label="raw EVI")
plt.ylim([-0.5, 1.2]);
ax.legend(loc="lower right");
# ax.set_title(a_field.CropTyp.unique()[0]);

# %% [markdown] id="zTYd9dqLSsxn"
# # Efficiency 
#
# Can we make this more efficient by doing the calculations in place as opposed to creating a new ```dataframe``` and copying stuff. Perhaps ```.map(.)``` too.
#
# **Remove outliers**

# %% id="bAHLlnE7Szjw"
reduced["ID"] = reduced["ID"].astype(str)
IDs = np.sort(reduced["ID"].unique())
VI_idx = "NDVI"


# %% colab={"base_uri": "https://localhost:8080/"} id="FAxGeCA-TEoF" outputId="8b052ec1-7a3e-4608-fd21-430b38824093"
no_outlier_df = pd.DataFrame(data = None,
                         index = np.arange(reduced.shape[0]), 
                         columns = reduced.columns)
counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = reduced[reduced["ID"]==a_poly].copy()
    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        ##************************************************
        #
        #    Set negative index values to zero.
        #
        ##************************************************
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = VI_idx)
        no_Outlier_TS.loc[no_Outlier_TS[VI_idx
                                        ] < 0 , VI_idx] = 0 

        """
        it is possible that for a field we only have x=2 data points
        where all the EVI/NDVI is outlier. Then, there is nothing to 
        use for interpolation. So, hopefully interpolate_outliers_EVI_NDVI is returning an empty data table.
        """ 
        if len(no_Outlier_TS) > 0:
            no_outlier_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]

# Sanity check. Will neved occur. At least should not!
no_outlier_df.drop_duplicates(inplace=True)

# %% [markdown] id="L0Bu6xyNTIN7"
# **Remove the jumps**
#
# Maybe we can remove old/previous dataframes to free memory up!

# %% colab={"base_uri": "https://localhost:8080/"} id="bn2-bHZ4Tfk9" outputId="8bdf4fdf-c31e-4a02-8cd6-2ac7edee5de5"
noJump_df = pd.DataFrame(data = None,
                         index = np.arange(no_outlier_df.shape[0]), 
                         columns = no_outlier_df.columns)
counter, row_pointer = 0, 0

for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = no_outlier_df[no_outlier_df["ID"]==a_poly].copy()
    
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################

    no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie = curr_field, 
                                                        give_col = VI_idx
                                                        , 
                                                        maxjump_perDay = 0.018)

    noJump_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

noJump_df['human_system_start_time'] = pd.to_datetime(noJump_df['human_system_start_time'])

# Sanity check. Will neved occur. At least should not!
print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape)) 
noJump_df.drop_duplicates(inplace=True)
print ("Shape of noJump_df after dropping duplicates is {}.".format(noJump_df.shape))

# %% id="mG6cq37a28cc"
del(no_Outlier_TS)

# %% [markdown] id="evXDiJKxTjEQ"
# **Regularize**
#
# Here we regularize the data. "Regularization" means pick a value for every 10-days. Doing this ensures 
#
# 1.   all inputs have the same length, 
# 2.   by picking maximum value of a VI we are reducing the noise in the time-series by eliminating noisy data points. For example, snow or shaddow can lead to understimating the true VI.
#
# Moreover, here, I am keeping only 3 columns. As long as we have ```ID``` we can
# merge the big dataframe with the final result later, here or externally.
# This will reduce amount of memory needed. Perhaps I should do this
# right the beginning.

# %% colab={"base_uri": "https://localhost:8080/"} id="fynzOVKCT3qE" outputId="05f1347d-dd03-4aff-cc55-7e9e85539c38"
# %%time

# define parameters
regular_window_size = 10
reg_cols = ['ID', 'human_system_start_time', VI_idx] # system_start_time list(noJump_df.columns)

st_yr = noJump_df.human_system_start_time.dt.year.min()
end_yr = noJump_df.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size)) # no_days // regular_window_size

nrows = no_steps * len(IDs)
print('st_yr is {}.'.format(st_yr))
print('end_yr is {}.'.format(end_yr))
print('nrows is {}.'.format(nrows))
print (long_eq)


regular_df = pd.DataFrame(data = None,
                         index = np.arange(nrows), 
                         columns = reg_cols)
counter, row_pointer = 0, 0

for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = noJump_df[noJump_df["ID"]==a_poly].copy()
    ################################################################
    # Sort by date (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    regularized_TS = nc.regularize_a_field(a_df = curr_field, \
                                           V_idks = VI_idx, \
                                           interval_size = regular_window_size,\
                                           start_year = st_yr, \
                                           end_year = end_yr)
    
    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = VI_idx)
    # if (counter == 0):
    #     print ("regular_df columns:",     regular_df.columns)
    #     print ("regularized_TS.columns", regularized_TS.columns)
    
    ################################################################
    # row_pointer = no_steps * counter
    
    """
       The reason for the following line is that we assume all years are 366 days!
       so, the actual thing might be smaller!
    """
    # why this should not work?: It may leave some empty rows in regular_df
    # but we drop them at the end.
    regular_df[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
    row_pointer += regularized_TS.shape[0]
    counter += 1

regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
regular_df.drop_duplicates(inplace=True)
regular_df.dropna(inplace=True)

# Sanity Check
regular_df.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
regular_df.reset_index(drop=True, inplace=True)

del(noJump_df)

# %% id="lqzNrGrET74H"

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="ykCINQmQU9xd" outputId="b4ca8a35-d0aa-47e1-8260-45cfbae9eb18"
#  Pick a field
a_field = regular_df[regular_df.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3), sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], 
        a_field[VI_idx], 
        linestyle='-', label=VI_idx, linewidth=3.5, color="dodgerblue", alpha=0.8)

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% [markdown] id="IjmgbKcDVTiw"
# **Savitzky-Golay Smoothing**

# %% colab={"base_uri": "https://localhost:8080/"} id="oVhXa8DwVegh" outputId="d06d772a-943a-4af4-a591-64299c4ce956"
# %%time
counter = 0
window_len, polynomial_order = 7, 3

for a_poly in IDs:
    if (counter % 300 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = regular_df[regular_df["ID"]==a_poly].copy()
    
    # Smoothen by Savitzky-Golay
    SG = scipy.signal.savgol_filter(curr_field[VI_idx].values, window_length=window_len, polyorder=polynomial_order)
    SG[SG > 1 ] = 1 # SG might violate the boundaries. clip them:
    SG[SG < -1 ] = -1
    regular_df.loc[curr_field.index, VI_idx] = SG
    counter += 1

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="7MwDu6q2Vpji" outputId="e0d5d43a-8cd7-4bd5-af35-90bd836de9ef"
#  Pick a field
an_ID = reduced.ID.unique()[0]
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], 
        a_field[VI_idx], 
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
        label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="SpyopmVQWiFD" outputId="6570297a-ada2-452d-ad0f-3ae9a73e5e09"
regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
# regular_df = pd.merge(regular_df, SF_data, on=['ID'], how='left') # we can do this later.
regular_df.reset_index(drop=True, inplace=True)
regular_df = nc.initial_clean(df=regular_df, column_to_be_cleaned = VI_idx
                              )
regular_df.head(2)

# %% [markdown] id="RCVxZ2CJtUkR"
# **Widen the data to use with ML**

# %% id="GwjpZeyCtdHB"
VI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37)]
columnNames = ["ID", "year"] + VI_colnames

years = regular_df.human_system_start_time.dt.year.unique()
IDs = regular_df.ID.unique()
no_rows = len(IDs) * len(years)

data_wide = pd.DataFrame(columns=columnNames, index=range(no_rows))
data_wide.ID = list(IDs) * len(years)
data_wide.sort_values(by=["ID"], inplace=True)
data_wide.reset_index(drop=True, inplace=True)
data_wide.year = list(years) * len(IDs)

for an_ID in IDs:
    curr_field = regular_df[regular_df.ID == an_ID]
    curr_years = curr_field.human_system_start_time.dt.year.unique()
    for a_year in curr_years:
        curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]

        data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index

        if VI_idx == "EVI":
            data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_field_year.EVI.values[:36]
        elif VI_idx == "NDVI":
            data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = curr_field_year.NDVI.values[:36]

# %% [markdown] id="AYMI0RwcWjc7"
# # Please tell me where to look for the trained models and I will make you happy!

# %% id="4HwojA6Ppui5"
model_dir = "/content/drive/MyDrive/NASA_trends/Models_Oct17/"


# %% [markdown] id="Y4HqLXrws4Da"
# # Functions we need
# We need the following two functions in case we want to use K-Nearest Neighbor or Deep Learning.
#
# -    Traditionnaly, images of dogs and cats are saved on the disk and read from there. Here I must try to figure out how to do them on the fly.

# %% id="XSOHwvICs8p5"
# We need this for KNN
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


# We need this for DL
# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img



# %% id="3iZ64rAvv6Uf"
model = "SVM"
winnerModel = "SG_NDVI_SVM_NoneWeight_00_Oct17_AccScoring_Oversample_SR3.sav"

# %% id="XVEyEblRwaI8"
import pickle

# %% id="cHr4PiBJwfQU"
if winnerModel.endswith(".sav"):
    # f_name = VI_idx + "_" + smooth + "_intersect_batchNumber" + batch_no + "_wide_JFD.csv"
    # data_wide = pd.read_csv(in_dir + f_name)
    # print("data_wide.shape: ", data_wide.shape)

    ML_model = pickle.load(open(model_dir + winnerModel, "rb"))
    predictions = ML_model.predict(data_wide.iloc[:, 2:])
    pred_colName = model + "_" + VI_idx  + "_preds"
    A = pd.DataFrame(columns=["ID", "year", pred_colName])
    A.ID = data_wide.ID.values
    A.year = data_wide.year.values
    A[pred_colName] = predictions
    predictions = A.copy()
    del A

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="QKBAJpkLwJLj" outputId="5a400bab-81ad-443a-a0e5-9449901a6670"
predictions = pd.merge(predictions, SF_data, on=['ID'], how='left')
predictions

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="k4JyjzKv43zL" outputId="d8b4c409-76ad-4852-d40b-551aaca1709e"
#  Pick a field
an_ID = "106054_WSDA_SF_2017"
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], 
        a_field[VI_idx], 
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
        label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% id="VKskUUEQ47Oz"
