#####
##### Sept 26 Sid found another eqs for computing nitrogen update
##### to see of the uptake is negative with that or not. Here we plot that
##### Here we are also plotting fields that are larger than 10 acres. 
##### about 150 fields are less than 10 acres
#####

import csv
import numpy as np
import pandas as pd

import datetime
from datetime import date
import time

import scipy
import scipy.signal
import os, os.path

from patsy import cr

# from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import ConciseDateFormatter

from datetime import datetime
register_matplotlib_converters()

import sys
start_time = time.time()

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

sys.path.append('/home/hnoorazar/NASA/')
import NASA_core as nc
import NASA_plot_core as ncp

####################################################################################
###
###      Parameters                   
###
####################################################################################
####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/data/hydro/users/Hossein/Sids_Projects/nitrogen/00_shapefiles/"
data_dir = "/data/hydro/users/Hossein/Sids_Projects/nitrogen/01_GEE_TS/"
plot_dir = "/data/hydro/users/Hossein/Sids_Projects/nitrogen/02_plots/"
print ("_________________________________________________________")
print ("data dir is: " + data_dir)
print ("_________________________________________________________")

####################################################################################
###
###                   Read data
###
####################################################################################
fName="Potato_Sent_Sept28_2ndFormula_2020-01-01_2021-01-01.csv"
potato = pd.read_csv(data_dir + fName)
meta=pd.read_csv(param_dir + "corn_potato_data2020.csv")

##### Drop small fields
potato = pd.merge(potato, meta[["ID", "ExactAcres"]], on=['ID'], how='left')
potato=potato[potato.ExactAcres>10].copy()

potato.dropna(subset=['NuptakeGEE'], inplace=True)
potato.reset_index(drop=True, inplace=True)

potato = nc.add_human_start_time_by_system_start_time(potato)

###### Round the numbers
# potato=potato.round(3)


potato_nit_min=potato.NuptakeGEE.min()
potato_nit_max=potato.NuptakeGEE.max()

size = 20
title_FontSize = 10
legend_FontSize = 14
tick_FontSize = 18
label_FontSize = 14

params = {'legend.fontsize': 17,
          'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.75,
          'ytick.labelsize': size * 0.75,
          'axes.titlepad': 10}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)

def plot_oneColumn_CropTitle_scatter(raw_dt, ax, titlee, idx="NDVI", 
                                     _label = "raw", _color="red",
                                     y_min=-1, y_max=1):

    ax.plot(raw_dt['human_system_start_time'], raw_dt[idx], c=_color, linewidth=2,
                label=_label);

    ax.set_title(titlee)
    ax.set_ylabel(idx, fontsize=20) # , labelpad=20); # fontsize = label_FontSize,
    ax.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    ax.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
    ax.legend(loc="upper right");
    # plt.yticks(np.arange(0, 1, 0.2))
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.set_ylim(y_min-0.1, y_max+0.1)


indeks="NuptakeGEE"
IDs=potato.ID.unique()

counter = 0
for _id in IDs:
    curr = potato[potato.ID == _id].copy()
    curr.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
    curr_meta = meta[meta.ID==_id]
    titlee = " ".join(curr_meta.CropTyp.unique()[0].split(", ")[::-1])

    fig, axs = plt.subplots(1, 1, figsize=(15, 4), sharex=False, sharey='col',
                            # sharex=True, sharey=True,
                           gridspec_kw={'hspace': 0.35, 'wspace': .05});
    axs.grid(True);
    
    # if curr.CropTyp.unique()[0]=="Corn, Field":
    #     curr_min=corn_F_min
    #     curr_max=corn_F_max
    # elif curr.CropTyp.unique()[0]=="Corn, Sweet":
    #     curr_min=corn_Sweet_min
    #     curr_max=corn_Sweet_max
    # elif curr.CropTyp.unique()[0]=="Corn Seed":
    #     curr_min=corn_Seed_min
    #     curr_max=corn_Seed_max
    # else:
    curr_min=potato_nit_min
    curr_max=potato_nit_max

    plot_oneColumn_CropTitle_scatter(raw_dt=curr, ax=axs, idx=indeks, titlee=titlee,
                                     _label = "Canopy N",
                                     _color="dodgerblue", 
                                     y_min=curr_min, y_max=curr_max)

    # save flat
    sub_dir = "".join("_".join(curr_meta.CropTyp.unique()[0].split(" ")).split(","))
    sub_dir = "/potato_Eq2_Sept26_"+sub_dir

    # "_".join("_".join(curr_meta.CropTyp.unique()[0].split(", ")[::-1]).split(" ")[::-1])
    # "_".join(curr_meta.CropTyp.unique()[0].split(", ")[::-1])
    plot_path = plot_dir + "/cornPotatopPlots/" + sub_dir + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + curr_meta.county.unique()[0] + "_" + _id +'.png'
    plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight', facecolor="w")

    plt.close('all')
    counter += 1


print ("done")

end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))


