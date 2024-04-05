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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# %%

# %%
# # %pylab inline --no-import-all
# # %autoreload

# %%
import numpy as np

# seed the pseudorandom number generator
from random import seed
from random import random


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize'] = 10

# %%
print('matplotlib: {}'.format(mpl.__version__))

# %%
plot_dir = "/Users/hn/Documents/01_research_data/Amin/"
    
with plt.xkcd():
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
    # axs.grid(axis="y", which="both")

    vec_len = 100
    xy = np.arange(-0.5, 11, 0.1);
    # minus_xy = np.arange(-0.5, 11, 0.1);

    np.random.seed(4)
    x_line = 10 * np.random.rand(vec_len);
    y_line = x_line + np.random.normal(loc=0, scale=1, size=vec_len)
    
    axs.scatter(x_line, y_line, s = 20, c="dodgerblue", marker="x");
    
    plt.xlim(-1, 12)
    plt.ylim(-1, 12)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    file_name = plot_dir + "orig_data_xkcd.pdf"
    plt.savefig(fname = file_name, dpi=100, bbox_inches='tight', transparent=False)
    plt.show()

##############
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
# axs.grid(axis="y", which="both")

vec_len = 100
xy = np.arange(-0.5, 11, 0.1);
# minus_xy = np.arange(-0.5, 11, 0.1);

np.random.seed(4)
x_line = 10 * np.random.rand(vec_len);
y_line = x_line + np.random.normal(loc=0, scale=1, size=vec_len)

axs.scatter(x_line, y_line, s = 20, c="dodgerblue", marker="x");

plt.xlim(-1, 12)
plt.ylim(-1, 12)

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

file_name = plot_dir + "orig_data.pdf"
plt.savefig(fname = file_name, dpi=100, bbox_inches='tight', transparent=False)
plt.show()

# %%
plot_dir = "/Users/hn/Documents/01_research_data/Amin/"
    
with plt.xkcd():
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
    # axs.grid(axis="y", which="both")

    vec_len = 100
    xy = np.arange(-0.5, 11, 0.1);
    # minus_xy = np.arange(-0.5, 11, 0.1);

    np.random.seed(4)
    x_line = 10 * np.random.rand(vec_len);
    y_line = x_line + np.random.normal(loc=0, scale=1, size=vec_len)
    
    axs.scatter(x_line, y_line, s = 20, c="dodgerblue", marker="x");
    axs.plot(xy, xy, color="r", linewidth=1, zorder=-1);
    
    plt.xlim(-1, 12)
    plt.ylim(-1, 12)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    file_name = plot_dir + "PCA_coord_1_xkcd.pdf"
    plt.savefig(fname = file_name, dpi=100, bbox_inches='tight', transparent=False)
    plt.show()

##############
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
# axs.grid(axis="y", which="both")

vec_len = 100
xy = np.arange(-0.5, 11, 0.1);
# minus_xy = np.arange(-0.5, 11, 0.1);

np.random.seed(4)
x_line = 10 * np.random.rand(vec_len);
y_line = x_line + np.random.normal(loc=0, scale=1, size=vec_len)

axs.scatter(x_line, y_line, s = 20, c="dodgerblue", marker="x");
axs.plot(xy, xy, color="r", linewidth=1, zorder=-1);

plt.xlim(-1, 12)
plt.ylim(-1, 12)

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

file_name = plot_dir + "PCA_coord_1.pdf"
plt.savefig(fname = file_name, dpi=100, bbox_inches='tight', transparent=False)
plt.show()

# %%
plot_dir = "/Users/hn/Documents/01_research_data/Amin/"
    
with plt.xkcd():
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
    # axs.grid(axis="y", which="both")

    vec_len = 100
    xy = np.arange(-0.5, 11, 0.1);
    # minus_xy = np.arange(-0.5, 11, 0.1);

    np.random.seed(4)
    x_line = 10 * np.random.rand(vec_len);
    y_line = x_line + np.random.normal(loc=0, scale=1, size=vec_len)
    
    axs.scatter(x_line, y_line, s = 20, c="dodgerblue", marker="x");
    axs.plot(xy, xy, color="r", linewidth=1, zorder=-1);

    minus_xy = xy[int(len(xy)/3) : int(2*len(xy)/3)]
    axs.plot(minus_xy, 10-minus_xy, color="r", linewidth=1, zorder=-1);
    
    plt.xlim(-1, 12)
    plt.ylim(-1, 12)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    file_name = plot_dir + "PCA_coord_xkcd.pdf"
    plt.savefig(fname = file_name, dpi=100, bbox_inches='tight', transparent=False)
    plt.show()

##############
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
# axs.grid(axis="y", which="both")

vec_len = 100
xy = np.arange(-0.5, 11, 0.1);
# minus_xy = np.arange(-0.5, 11, 0.1);

np.random.seed(4)
x_line = 10 * np.random.rand(vec_len);
y_line = x_line + np.random.normal(loc=0, scale=1, size=vec_len)

axs.scatter(x_line, y_line, s = 20, c="dodgerblue", marker="x");
axs.plot(xy, xy, color="r", linewidth=1, zorder=-1);

minus_xy = xy[int(len(xy)/3) : int(2*len(xy)/3)]
axs.plot(minus_xy, 10-minus_xy, color="r", linewidth=1, zorder=-1);

plt.xlim(-1, 12)
plt.ylim(-1, 12)

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

file_name = plot_dir + "PCA_coord.pdf"
plt.savefig(fname = file_name, dpi=100, bbox_inches='tight', transparent=False)
plt.show()

# %%
import random
random.uniform(-1, 1)

# %%

# %%
# seed random number generator
seed(1)
print(random())

seed(1)
print(random())

# %%
np.random.seed(1)
print (np.random.rand(10))

# %%
with plt.xkcd():
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
    # axs.grid(axis="y", which="both")

    vec_len = 100
    xy = np.arange(-0.5, 11, 0.1);
    # minus_xy = np.arange(-0.5, 11, 0.1);

    np.random.seed(4)
    x_line = 10 * np.random.rand(vec_len);
    y_line = x_line

    # axs.plot(xy, 5 * np.ones(len(xy)), color="r", linewidth=1);
    plt.axhline(y = 5, color = 'r', linestyle = '-', zorder=-1) 
    axs.scatter(x_line, 5 * np.ones(len(x_line)), s = 20, c="dodgerblue", marker="x");

    # axs.set_axis_off()
    plt.box(True)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.xlim(-1, 11)
    plt.ylim(4.5, 5.5)

    file_name = plot_dir + "PCA_coord_dim1_xkcd.pdf"
    plt.savefig(fname = file_name, dpi=100, bbox_inches='tight', transparent=False)
    plt.show()

#####

fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
# axs.grid(axis="y", which="both")

vec_len = 100
xy = np.arange(-0.5, 11, 0.1);
# minus_xy = np.arange(-0.5, 11, 0.1);

np.random.seed(4)
x_line = 10 * np.random.rand(vec_len);
y_line = x_line

# axs.plot(xy, 5 * np.ones(len(xy)), color="r", linewidth=1, zorder=-1);
plt.axhline(y = 5, color = 'r', linestyle = '-', zorder=-1)
axs.scatter(x_line, 5 * np.ones(len(x_line)), s = 20, c="dodgerblue", marker="x");

# axs.set_axis_off()
# plt.box(True)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.xlim(-1, 11)
plt.ylim(4.5, 5.5)

file_name = plot_dir + "PCA_coord_dim1.pdf"
plt.savefig(fname = file_name, dpi=100, bbox_inches='tight', transparent=False)
plt.show()

# %%
