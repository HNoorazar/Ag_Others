# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
# #### This tutorial is from [this source](https://github.com/scikit-tda/scikit-tda/blob/master/docs/notebooks/scikit-tda%20Tutorial.ipynb).
#
#
# #### scikit-tda modules
#    - **Kepler Mapper** — mapper algorithm
#    - **Ripser** — persistence homology computations
#    - **Persim** — comparison and analysis of persistence diagrams
#    - **CechMate** — advanced and custom filtrations
#    - **TDAsets** — synthetic data sets of some manifolds (shperes, torus, swiss rolls)
#    
# There is another package too: [https://github.com/MathieuCarriere/sklearn-tda](https://github.com/MathieuCarriere/sklearn-tda).
# They have the TDA clustering algorithms as well.

# %%
# plt.suptitle(f"{norm(u)=:.4g}");

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
import tadasets
import persim
# # !pip3 install ripser
# pip install --upgrade numpy
import ripser

# %%
# PV = !python --version
print ("!python --version is '{}'.".format(PV));

print ("np.__version__ is '{}'.".format(np.__version__));

import sys;
print ("sys.version is '{}'.".format(sys.version));

# %% [markdown]
# #### 1. Persistence API
# At first, let's generate a toy example. ```tadasets``` has few functions to generate random point clouds of different shapes. Below we generate two samples of size ```n=100``` from 1-spheres:
#
#    - ```data_clean``` data is sampled directly from a unit cirle
#    - ```data_noisy``` data is sampled from a unit circle with added standard normal ```noise``` scaled by noise parameter.

# %%
import tadasets
np.random.seed(565656)

data_clean = tadasets.dsphere(d=1, n=100, noise=0.0)
data_noisy = tadasets.dsphere(d=1, n=100, noise=0.10) 

# data_clean = tadasets.infty_sign(n=100, noise=0.0)
# data_noisy = tadasets.infty_sign(n=100, noise=0.15) 

plt.rcParams["figure.figsize"] = (6, 6)
plt.scatter(data_clean[:,0], data_clean[:,1], label="clean data", s=8)
plt.scatter(data_noisy[:,0], data_noisy[:,1], label="noisy data", s=16, alpha=0.6)
plt.axis('equal')

plt.legend()
plt.show()

# %%
size = 10
title_FontSize = 5
legend_FontSize = 4
tick_FontSize = 6
label_FontSize = 7

params = {'legend.fontsize': size, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size, #  * 0.75
          'ytick.labelsize': size, #  * 0.75
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

# %%
ripser.ripser(data_clean).keys()

# %%
import math
variable = math.pi
print(f"Using Numeric {variable =:0.4G}")
print(f"|{variable:30}|")
print(f"|{variable:<25}|")
print(f"|{variable:=<25}|")

print(f"|{variable:>25}|")
print(f"|{variable:=>100}|")

print(f"|{variable:^25}|")
print(f"{variable:=^100}")

# %%
variable_line = "="
print (f"{ripser.ripser(data_clean)['num_edges']=}")
print(f"{variable_line:=^100}")
print ('ripser.ripser(data_clean)["dperm2all"] is \n', ripser.ripser(data_clean)["dperm2all"] )
print(f"{variable_line:=^100}")
print ('ripser.ripser(data_clean)["idx_perm"] is \n', ripser.ripser(data_clean)["idx_perm"])
print(f"{variable_line:=^100}")
print (f'{ripser.ripser(data_clean)["r_cover"]=}')


# %%

# %% [markdown]
# #### 1.1. Computing and ploting persistence diagrams for point cloud data
# To compute the PD of a point cloud we are using ```ripser``` module. Either directly or via an object of Rips class
# To plot PDs we use ```persim``` module.
# ```ripser``` can take a distance matrix as input
#
# #### 1.1.1. Out of the box persistence

# %%
def diagram_sizes(dgms):
    return ", ".join([f"|$H_{i}$|={len(d)}" for i, d in enumerate(dgms)])


# %%
dgm_clean = ripser.ripser(data_clean)['dgms']
persim.plot_diagrams(dgm_clean, 
                     show=True, 
                     title=f"Clean\n{diagram_sizes(dgm_clean)}"
                    )

# %% [markdown]
# ### 1.1.2. Class interface
# The **Ripser** module also provides a sklearn-style class ```Rips```.

# %%
from ripser import Rips
rips = Rips()
dgm_noisy = rips.transform(data_noisy)
rips.plot(show=True, title=f"Noisy\n{diagram_sizes(dgm_noisy)}");

# %%
print ("dgm_clean[0].shape is {}.".format(dgm_clean[0].shape))
print ("dgm_clean[1].shape is {}.".format(dgm_clean[1].shape))
print ("___________________________________________________________")
print ("dgm_noisy[0].shape is {}.".format(dgm_noisy[0].shape))
print ("dgm_noisy[1].shape is {}.".format(dgm_noisy[1].shape))

# %% [markdown]
# # The following 2 do the same thing:

# %%
ripser_ripser_clean = ripser.ripser(data_clean)['dgms']
rips_transform_clean = rips.transform(data_clean)

print ((ripser_ripser_clean[0]==rips_transform_clean[0]).sum())
(ripser_ripser_clean[1]==rips_transform_clean[1])

# %%
data_clean.shape

# %%

# %%
# plt.plot(dgm_noisy_0_x, dgm_noisy_0_y, 'g^', dgm_noisy_1_x, dgm_noisy_1_y, 'bs');
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.15, 'wspace': .15});
(ax1, ax2) = axs;
ax1.grid(True); ax2.grid(True)

scatter_size=5
ax1.scatter(dgm_clean[0][:, 0], dgm_clean[0][:, 1], s=scatter_size, c="r", label="first array")
ax1.scatter(dgm_clean[1][:, 0], dgm_clean[1][:, 0], s=scatter_size, c="k", label="second array")
ax1.set_title("Clean")
ax1.legend(loc="lower right");


ax2.scatter(dgm_noisy[0][:, 0], dgm_noisy[0][:, 1], s=scatter_size, c="r", label="first array")
ax2.scatter(dgm_noisy[1][:, 0], dgm_noisy[1][:, 0], s=scatter_size, c="k", label="second array")
ax2.set_title("Noisy")
ax2.legend(loc="lower right");
# ax.scatter(dgm_noisy_0_x, dgm_noisy_0_y, s=10, c="r", marker="^", label="noisy first array")
# ax.scatter(dgm_noisy_1_x, dgm_noisy_1_y, s=10, c="k", marker="^", label="noisy second array")
# ax.legend(loc="best");
# ax.set_ylim(0, 1.05)

# %%

# %%
# plt.plot(dgm_noisy_0_x, dgm_noisy_0_y, 'g^', dgm_noisy_1_x, dgm_noisy_1_y, 'bs');
# plt.scatter(dgm_noisy[0][:, 0], dgm_noisy[0][:, 1], s=10, c="r", label="noisy first array")
# plt.scatter(dgm_noisy[1][:, 0], dgm_noisy[1][:, 0], s=10, c="k", label="noisy second array")
# plt.legend(loc="best");
# ax.set_ylim(0, 1.05)

# %% [markdown]
# #### 1.1.3. Input option: Distance matrix
# An important feature of Ripser is its ability to take distance matrices as input. This flexiblity allwos Ripser to handle n-dimensional point clouds as well as more abstract metric spaces, for example, given by graphs

# %%
from sklearn.metrics.pairwise import pairwise_distances

D = pairwise_distances(data_noisy, metric='euclidean')

dgm_noisy = ripser.ripser(D, distance_matrix=True)['dgms']
persim.plot_diagrams(
    dgm_noisy, show=True, 
    title=f"Noisy\n{diagram_sizes(dgm_noisy)}"
) 

# %% [markdown]
# #### 1.2. Ripser's options
#    - ```maxdim: int``` — maximum homology dimension computed
#    - ```thresh: float``` — maximum radius for Rips filtration
#    - ```coeff: int``` — field of coefficients of homology
#
#
# ####  1.2.1. Maximum homology dimension
#    - 0 — points, connected components
#    - 1 — line segments, holes
#    - 2 — triangles, cavities

# %%
dgm_noisy = ripser.ripser(data_noisy, maxdim=2)['dgms']
persim.plot_diagrams(
    dgm_noisy, show=True, 
    title=f"Noisy: maxdim=2\n{diagram_sizes(dgm_noisy)}"
) 

# %% [markdown]
# #### 1.2.2. Maximum Radius for Rips filtration
#
# **Vietoris–Rips complex**, also called the Vietoris complex or Rips complex, is a way of forming a topological space from distances in a set of points. It is an abstract simplicial complex that contains a simplex for every finite set of points that has diameter at most 
# :
#
#    - if a finite set $S$ of $k$ points has the property that the distance between every pair of points in $S$ is at most $R$, then we include $S$ as a $(k-1)$-simplex in the complex.

# %%
thresh = 0.5
dgm_noisy = ripser.ripser(data_noisy, thresh=thresh)['dgms']
persim.plot_diagrams(dgm_noisy, show=True, 
                     title=f"Noisy: thresh={thresh}\n{diagram_sizes(dgm_noisy)}"
                    )

# %%

# %%
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def plot_rips_complex(data, R, label="data", col=1, maxdim=2):
    tab10 = cm.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(label)
    ax.scatter(
        data[:, 0], data[:, 1], label=label,
        s=8, alpha=0.9, c=np.array(tab10([col] * len(data)))
    )

    for xy in data:
        ax.add_patch(mpatches.Circle(xy, radius=R, fc='none', ec=tab10(col), alpha=0.2))

    for i, xy in enumerate(data):
        if maxdim >=1:
            for j in range(i + 1, len(data)):
                pq = data[j]
                if (xy != pq).all() and (np.linalg.norm(xy - pq) <= R):
                    pts = np.array([xy, pq])
                    ax.plot(pts[:, 0], pts[:, 1], color=tab10(col), alpha=0.6, linewidth=1)
                if maxdim == 2:
                    for k in range(j + 1, len(data)):
                        ab = data[k]
                        if ((ab != pq).all()
                                and (np.linalg.norm(xy - pq) <= R)
                                and (np.linalg.norm(xy - ab) <= R)
                                and (np.linalg.norm(pq - ab) <= R)
                        ):
                            pts = np.array([xy, pq, ab])
                            ax.fill(pts[:, 0], pts[:, 1], facecolor=tab10(col), alpha=0.1)
                        pass

    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    pass

plot_rips_complex(data_noisy, R=0.29, label="Noisy Data", maxdim=1)

# %% [markdown]
# #### 1.2.3. Ripser's options for the class interface
# When an object of the ```Rips``` class is used, the options are set as fields of the ripser object.

# %%
rips = Rips()
rips.maxdim = 2
rips.thresh = 0.5
rips.coef = 3
rips.transform(data_noisy)
rips.plot(show=True, title='Noisy');

# %% [markdown]
# #### 1.3. Plotting options
#    - persistence diagrams of different data sets on one plot
#    - customizing the plot:, ```xy_range```, ```title: str```, ```size: str```, 
#    - ```lifetime```
#    - persistence images
# ####  1.3.1. PDs of different data sets on one plot

# %%
dgms_clean = ripser.ripser(data_clean)['dgms']
dgms_noisy = ripser.ripser(data_noisy, maxdim=2)['dgms']
persim.plot_diagrams([dgms_clean[1], dgms_noisy[1]], 
                     labels=['Clean $H_1$', 'Noisy $H_1$'], 
                     show=True
                    )

# %% [markdown]
# #### 1.3.2. Customizing plot

# %%
persim.plot_diagrams(
    dgms_noisy, 
    title="Noisy Data",
    labels=["noisy $H_0$", "noisy $H_1$"],
    colormap="bmh",
    xy_range=[-2,3, -1, 2],
    size=10,
    diagonal=True,
    show=True
)

# %% [markdown]
# #### 1.3.3. Lifetime plots

# %%
persim.plot_diagrams(
    dgms_noisy, 
    title="Noisy Data",
    lifetime=True,
    show=True
)

# %% [markdown]
# #### 1.3.4. Persistence images
# The **Persim** module contains class ```PersistenceImager``` that is a transformer which converts persistence diagrams into persistence images.

# %%
from persim import PersistenceImager

pimager = PersistenceImager(pixel_size=0.2)

# The `fit()` method can be called on one or more (*,2) numpy arrays 
# to automatically determine the miniumum birth and persistence ranges needed to capture all persistence pairs. 
# The ranges and resolution are automatically adjusted to accomodate the specified pixel size.
pimager.fit(dgms_noisy[1:3])

# The `transform()` method can then be called on one or more (*,2) numpy arrays 
# to generate persistence images from diagrams.
imgs = pimager.transform(dgms_noisy[1:3])

fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].set_title("Original Diagram")
persim.plot_diagrams(dgms_noisy[1:3],  lifetime=True, ax=axs[0], labels=["$H_1$", "$H_2$"])

axs[1].set_title("Persistence Image")
pimager.plot_image(imgs[0], ax=axs[1])


# %% [markdown]
# #### 2. Analysis of PDs
# #### 2.1. Distance between diagrams
#
# **persim** package contains implementations of a number of distance functions between persistence diagrams:
#
#    - Wasserstein distance with matching between persistence diagrams.
#    - Bottleneck distance with matching between persistence diagrams.
#    - Estimation of Gromov-Hausdorff distance.
#    - The pseudo-metric between two diagrams based on the continuous heat kernel.

# %%
dgm_clean = ripser.ripser(data_clean)['dgms'][1]
dgm_noisy = ripser.ripser(data_noisy)['dgms'][1]

# The bottleneck of the matching is shown as a red line, 
# while the other pairs in the perfect matching which are less than the diagonal are shown as green lines 
# (NOTE: There may be many possible matchings with the minimum bottleneck, and this returns an arbitrary one)
distance_bottleneck, matching = persim.bottleneck(dgm_clean, dgm_noisy, matching=True)
persim.bottleneck_matching(
    dgm_clean, dgm_noisy, matching, 
    labels=['Clean $H_1$', 'Noisy $H_1$']
)
plt.title(f"Bottleneck distance = {distance_bottleneck:0.4f}",)
plt.show()


# %% [markdown]
# #### 2.2. Persistence Images in Classification
# We construct datasets from two classes, one just noise and the other noise with two circles in the middle.
# We then compute persistence diagrams with **ripser**.
# In order to apply ML algorithms, we need to vectorize computed PDs. We do this by converting them into persistence images with **Persim**.
# Using these persistence images, we build a Logistic Regression model using a LASSO penatly to classify whether the dataset has a circle or not.
#
# #### 2.2.1. Data
# We start by generating $M$ point clouds each of size $N$: half of them are noise, another half are noise with circles.

# %%
np.random.seed(565656)
M = 50           # total number of samples
m = int(M / 2)   # number of samples per class ('noise'/'circles')
N = 400          # number of points per dataset

def noise(N, scale):
    return scale * np.random.random((N, 2))

def circle(N, scale, offset):
    """Generates two circles with center at `offset` scaled by `scale`"""
    half = int(N/2)
    circ = np.concatenate(
        (tadasets.dsphere(d=1, n=half, r=1.1, noise=0.05),
        tadasets.dsphere(d=1, n=N-half, r=0.4, noise=0.05))
    )
    return offset + scale * circ

# Generate data
just_noise = [noise(N, 150) for _ in range(m)]

half = int(N / 2)
with_circle = [np.concatenate((circle(half, 50, 70), noise(N - half, 150)))
               for _ in range(m)]

datas = []
datas.extend(just_noise)
datas.extend(with_circle)

# Define labels
labels = np.zeros(M)
labels[m:] = 1

# Visualize the data
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(8,4)

xs, ys = just_noise[0][:,0], just_noise[0][:,1]
axs[0].scatter(xs, ys, s=10)
axs[0].set_title("Example noise dataset")
axs[0].set_aspect('equal', 'box')

xs_, ys_ = with_circle[0][:,0], with_circle[0][:,1]
axs[1].scatter(xs_, ys_, s=10)
axs[1].set_title("Example noise with circle dataset")
axs[1].set_aspect('equal', 'box')

fig.tight_layout()

# %% [markdown]
# #### 2.2.2. Persistence Diagrams
# For each point cloud we generate a $H_1$ persistence diagramm.

# %%
rips = ripser.Rips(maxdim=1, coeff=2)
diagrams_h1 = [rips.fit_transform(data)[1] for data in datas]

plt.figure(figsize=(8,4))

plt.subplot(121)
rips.plot(diagrams_h1[0], show=False, lifetime=True)
plt.title("PD of $H_1$ for just noise")

plt.subplot(122)
rips.plot(diagrams_h1[-1], show=False, lifetime=True)
plt.title("PD of $H_1$ for circle w/ noise")

plt.show()

# %% [markdown]
# #### 2.2.3. Persistence Images
# Next, each persistence diagram is turned into a persistence image, which is just a 2d-array of pixels.

# %%
pimgr = PersistenceImager(pixel_size=0.8)
pimgr.fit(diagrams_h1)
imgs = pimgr.transform(diagrams_h1)
print(f"PI Resolution = {pimgr.resolution}")

plt.figure(figsize=(12, 7))
ax = plt.subplot(121)
pimgr.plot_image(imgs[0], ax)
plt.title("PI of $H_1$ for noise")

ax = plt.subplot(122)
pimgr.plot_image(imgs[-1], ax)
plt.title("PI of $H_1$ for circle w/ noise")

# %% [markdown]
# #### 2.2.4. Classification
# Now, we flatten 2d-images into 1d-arrays and randomly split them into test and traing data sets.
#
# Finally, we fit Logistic Regression with a LASSO penatly to the training data, and compute mean accuracy on the test data.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# train/test data
imgs_array = np.array([img.flatten() for img in imgs])
X_train, X_test, y_train, y_test = train_test_split(
    imgs_array, labels, test_size=0.40, random_state=42
)
print(f"Train size = {X_train.shape[0]}\n"
      f"Test size  = {X_test.shape[0]} \n"
      f"Dimensions = {imgs_array.shape[1]}")

# logistic regression
lr = LogisticRegression(penalty='l1', solver='liblinear')
lr.fit(X_train, y_train)
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)
print(f"{'-'*35}\nTrain score = {train_score}; Test score={test_score}")

# %% [markdown]
# #### 3. Persistence for graphs
#
# #### 3.1. Species-reaction graph data
# The graph below is a species-reaction graph and it represents an atmospheric chemical mechanisms, in particular the Super-Fast chemical mechanism (SF).
#
# There are 18 chemicals and 20 reactions. It is a **bipartite** graph.
# Weights are linearly related to (log) rates of reactions and are **positive**.
# The graph is **directed**, but we will transform it into an undirected graph.
#
# ####  3.1.1. Reading graph data

# %%
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

# reading nodes
nodes = pd.read_csv(
    "data/sfNames.csv", 
    header=0, sep=" ", 
    dtype={'id': int, 'name': str}, 
)

nodes['isRxn'] = nodes['name'].str.contains("Rxn").astype(int)
nodes["name"] = nodes['name'].str.replace("Rxn: ", "").str.replace(" ", "")
nodes["color"] = pd.Series(pd.Categorical.from_codes(nodes["isRxn"], categories=["cyan", "orange"]))
nodes = nodes.set_index('id', drop=False)

# reading edges
edges = pd.read_csv(
    "data/sfEdgeWeights.csv", 
    header=0, sep=" ", 
    dtype=dict(source=int, target=int, weight=float), 
)

# create a bipartite graph
G = nx.from_pandas_edgelist(edges, create_using=nx.DiGraph, edge_attr=True,)
nx.set_node_attributes(G, nodes['isRxn'].to_dict(), "bipartite")
print(f"is bipartite = {bipartite.is_bipartite(G)}")

# %%
VI_idx = "V"
smooth_type = "N"
SR = 3
print (f"Passed Args. are: {VI_idx=:}, {smooth_type=:}, and {SR=:}!")

# %%
import pandas as pd
f_name = "KNN_SG_EVI_Oct17_AccScoring_Oversample_SR6_test_result.csv"
A = pd.read_csv("/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/KNN_results/overSample/" + f_name)

f_name = "KNN_SG_EVI_Oct17_AccScoring_Oversample_SR8_test_result.csv"
B = pd.read_csv("/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/KNN_results/overSample/" + f_name)

A.equals(B)

# %%
A = pd.read_csv("/Users/hn/Documents/observed_start_DoY_90_days2maturity_EE.csv")
A.head(2)

# %%

# %%
