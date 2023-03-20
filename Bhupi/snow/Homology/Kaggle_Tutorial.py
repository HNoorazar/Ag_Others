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

# %% [markdown]
# Kaggle tutorial actually did not have no clustering. So, name of this notebook must change

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
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

from pylab import imshow
import pickle
import h5py
import sys

# %%
# # !pip3 install ripser
# pip install --upgrade numpy
# # !pip3 install tadasets
# # !pip3 install kmapper
import ripser
from ripser import Rips #, ripser

import persim
# from persim import plot_diagrams

import tadasets

# Import the class
import kmapper as km


# %%
def diagram_sizes(dgms):
    return ", ".join([f"|$H_{i}$|={len(d)}" for i, d in enumerate(dgms)])


# %%
from sklearn.datasets import make_circles, make_blobs, make_moons, load_wine, load_breast_cancer, load_iris
import random

# %%
# https://www.kaggle.com/code/mikolajbabula/topological-approach-to-clustering/notebook
n_samples=100
noise=0.1
data, labels = make_moons(n_samples, noise=noise, shuffle=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.set_xlim([np.min(data[:,0])-.2, np.max(data[:,0])+.2])
ax1.set_ylim([np.min(data[:,1])-.2, np.max(data[:,1])+.2])
ax1.scatter(data[:,0], data[:,1], c=labels, cmap=plt.get_cmap('winter'))
ax1.axis('off')

# %%
# https://github.com/topolearn/topo-clustering/blob/main/top_clustering.py

import sys
import math
import random
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.cluster import contingency_matrix
from matplotlib import pyplot as plt


class TopClustering:
    """Topological clustering.
    
    Attributes:
        n_clusters: 
          The number of clusters.
        top_relative_weight:
          Relative weight between the geometric and topological terms.
          A floating point number between 0 and 1.
        max_iter_alt:
          Maximum number of iterations for the topological clustering.
        max_iter_interp:
          Maximum number of iterations for the topological interpolation.
        learning_rate:
          Learning rate for the topological interpolation.
        
    Reference:
        Songdechakraiwut, Tananun, Bryan M. Krause, Matthew I. Banks, Kirill V. Nourski, and Barry D. Van Veen. 
        "Fast topological clustering with Wasserstein distance." 
        International Conference on Learning Representations (ICLR). 2022.
    """

    def __init__(self, n_clusters, top_relative_weight, max_iter_alt,
                 max_iter_interp, learning_rate):
        self.n_clusters = n_clusters
        self.top_relative_weight = top_relative_weight
        self.max_iter_alt = max_iter_alt
        self.max_iter_interp = max_iter_interp
        self.learning_rate = learning_rate

    def fit_predict(self, data):
        """Computes topological clustering and predicts cluster index for each sample.
        
            Args:
                data:
                  Training instances to cluster.
                  
            Returns:
                Cluster index each sample belongs to.
        """
        data = np.asarray(data)
        n_node = data.shape[1]
        n_edges = math.factorial(n_node) // math.factorial(2) // math.factorial(
            n_node - 2)  # n_edges = (n_node choose 2)
        n_births = n_node - 1
        self.weight_array = np.append(
            np.repeat(1 - self.top_relative_weight, n_edges),
            np.repeat(self.top_relative_weight, n_edges))

        # Networks represented as vectors concatenating geometric and topological info
        X = []
        for adj in data:
            X.append(self._vectorize_geo_top_info(adj))
        X = np.asarray(X)

        # Random initial condition
        self.centroids = X[random.sample(range(X.shape[0]), self.n_clusters)]

        # Assign the nearest centroid index to each data point
        assigned_centroids = self._get_nearest_centroid(
            X[:, None, :], self.centroids[None, :, :])
        prev_assigned_centroids = assigned_centroids

        for it in range(self.max_iter_alt):
            for cluster in range(self.n_clusters):
                # Previous iteration centroid
                prev_centroid = np.zeros((n_node, n_node))
                prev_centroid[np.triu_indices(
                    prev_centroid.shape[0],
                    k=1)] = self.centroids[cluster][:n_edges]

                # Determine data points belonging to each cluster
                cluster_members = X[assigned_centroids == cluster]

                # Compute the sample mean and top. centroid of the cluster
                cc = cluster_members.mean(axis=0)
                sample_mean = np.zeros((n_node, n_node))
                sample_mean[np.triu_indices(sample_mean.shape[0],
                                            k=1)] = cc[:n_edges]
                top_centroid = cc[n_edges:]
                top_centroid_birth_set = top_centroid[:n_births]
                top_centroid_death_set = top_centroid[n_births:]

                # Update the centroid
                try:
                    cluster_centroid = self._top_interpolation(
                        prev_centroid, sample_mean, top_centroid_birth_set,
                        top_centroid_death_set)
                    self.centroids[cluster] = self._vectorize_geo_top_info(
                        cluster_centroid)
                except:
                    print(
                           'Error: Possibly due to the learning rate is not within appropriate range.'
                         )
                    sys.exit(1)

            # Update the cluster membership
            assigned_centroids = self._get_nearest_centroid(X[:, None, :], self.centroids[None, :, :])

            # Compute and print loss as it is progressively decreasing
            loss = self._compute_top_dist(
                X, self.centroids[assigned_centroids]).sum() / len(X)
            print('Iteration: %d -> Loss: %f' % (it, loss))

            if (prev_assigned_centroids == assigned_centroids).all():
                break
            else:
                prev_assigned_centroids = assigned_centroids
        return assigned_centroids

    def _vectorize_geo_top_info(self, adj):
        birth_set, death_set = self._compute_birth_death_sets(adj)  # topological info
        vec = adj[np.triu_indices(adj.shape[0], k=1)]  # geometric info
        return np.concatenate((vec, birth_set, death_set), axis=0)

    def _compute_birth_death_sets(self, adj):
        """Computes birth and death sets of a network."""
        mst, nonmst = self._bd_demomposition(adj)
        birth_ind = np.nonzero(mst)
        death_ind = np.nonzero(nonmst)
        return np.sort(mst[birth_ind]), np.sort(nonmst[death_ind])

    def _bd_demomposition(self, adj):
        """Birth-death decomposition."""
        eps = np.nextafter(0, 1)
        adj[adj == 0] = eps
        adj = np.triu(adj, k=1)
        Xcsr = csr_matrix(-adj)
        Tcsr = minimum_spanning_tree(Xcsr)
        mst = -Tcsr.toarray()  # reverse the negative sign
        nonmst = adj - mst
        return mst, nonmst

    def _get_nearest_centroid(self, X, centroids):
        """Determines cluster membership of data points."""
        dist = self._compute_top_dist(X, centroids)
        nearest_centroid_index = np.argmin(dist, axis=1)
        return nearest_centroid_index

    def _compute_top_dist(self, X, centroid):
        """Computes the pairwise top. distances between networks and centroids."""
        return np.dot((X - centroid)**2, self.weight_array)

    def _top_interpolation(self, init_centroid, sample_mean,
                           top_centroid_birth_set, top_centroid_death_set):
        """Topological interpolation."""
        curr = init_centroid
        for _ in range(self.max_iter_interp):
            # Geometric term gradient
            geo_gradient = 2 * (curr - sample_mean)

            # Topological term gradient
            sorted_birth_ind, sorted_death_ind = self._compute_optimal_matching(
                curr)
            top_gradient = np.zeros_like(curr)
            top_gradient[sorted_birth_ind] = top_centroid_birth_set
            top_gradient[sorted_death_ind] = top_centroid_death_set
            top_gradient = 2 * (curr - top_gradient)

            # Gradient update
            curr -= self.learning_rate * (
                (1 - self.top_relative_weight) * geo_gradient +
                self.top_relative_weight * top_gradient)
        return curr

    def _compute_optimal_matching(self, adj):
        mst, nonmst = self._bd_demomposition(adj)
        birth_ind = np.nonzero(mst)
        death_ind = np.nonzero(nonmst)
        sorted_temp_ind = np.argsort(mst[birth_ind])
        sorted_birth_ind = tuple(np.array(birth_ind)[:, sorted_temp_ind])
        sorted_temp_ind = np.argsort(nonmst[death_ind])
        sorted_death_ind = tuple(np.array(death_ind)[:, sorted_temp_ind])
        return sorted_birth_ind, sorted_death_ind


#############################################
################### Demo ####################
#############################################
def random_modular_graph(d, c, p, mu, sigma):
    """Simulated modular network.
    
        Args:
            d: Number of nodes.
            c: Number of clusters/modules.
            p: Probability of attachment within module.
            mu, sigma: Used for random edge weights.
            
        Returns:
            Adjacency matrix.
    """
    adj = np.zeros((d, d))  # adjacency matrix
    for i in range(1, d + 1):
        for j in range(i + 1, d + 1):
            module_i = math.ceil(c * i / d)
            module_j = math.ceil(c * j / d)

            # Within module
            if module_i == module_j:
                if random.uniform(0, 1) <= p:
                    w = np.random.normal(mu, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
                else:
                    w = np.random.normal(0, sigma)
                    adj[i - 1][j - 1] = max(w, 0)

            # Between modules
            else:
                if random.uniform(0, 1) <= 1 - p:
                    w = np.random.normal(mu, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
                else:
                    w = np.random.normal(0, sigma)
                    adj[i - 1][j - 1] = max(w, 0)
    return adj


def purity_score(labels_true, labels_pred):
    mtx = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(mtx, axis=0)) / np.sum(mtx)


def main():
    np.random.seed(0)
    random.seed(0)

    # Generate a dataset comprising simulated modular networks
    dataset = []
    labels_true = []
    n_network = 20
    n_node = 60
    p = 0.7
    mu = 1
    sigma = 0.5
    for module in [2, 3, 5]:
        for _ in range(n_network):
            adj = random_modular_graph(n_node, module, p, mu, sigma)
            # Uncomment lines below for visualization
            # plt.imshow(adj, vmin=0, vmax=2, cmap='YlOrRd')
            # plt.colorbar()
            # plt.show()
            dataset.append(adj)
            labels_true.append(module)

    # Topological clustering
    n_clusters = 3
    top_relative_weight = 0.99  # 'top_relative_weight' between 0 and 1
    max_iter_alt = 300
    max_iter_interp = 300
    learning_rate = 0.05
    print('Topological clustering\n----------------------')
    labels_pred = TopClustering(n_clusters, top_relative_weight, max_iter_alt,
                                max_iter_interp,
                                learning_rate).fit_predict(dataset)
    print('\nResults\n-------')
    print('True labels:', np.asarray(labels_true))
    print('Pred indices:', labels_pred)
    print('Purity score:', purity_score(labels_true, labels_pred))


if __name__ == '__main__':
    main()

# %% [markdown]
# # KeplerMapper?

# %%
snow_TS_dir_base = "/Users/hn/Documents/data/EithyYearsClustering/"
snow_TS_dir_base = "/Users/hn/Documents/01_research_data/Bhupi/snow/EithyYearsClustering/"
diff_dir = snow_TS_dir_base + "Brightness_temperature/Only_for_SNOTEL_grids/"

# %%
all_locs_all_years_but_2003 = pd.read_csv(snow_TS_dir_base + \
                                          "Brightness_temperature/" + \
                                          "all_locs_all_years_but_2003.csv")
all_locs_all_years_but_2003.date = pd.to_datetime(all_locs_all_years_but_2003.date)
all_locs_all_years_but_2003.head(2)

# %%
all_locs_after_2004 = all_locs_all_years_but_2003[all_locs_all_years_but_2003.year>=2004].copy()
all_locs_after_2004.reset_index(drop=True, inplace=True)

# %%
locations = list(all_locs_after_2004.columns)

bad_cols = ["month", "day", "year", "date"]
for a_bad in bad_cols:
    locations.remove(a_bad)

print (f"{len(locations)=}")

# Check if our data is daily
print (len(all_locs_after_2004))
print ((all_locs_after_2004.year.max()-all_locs_after_2004.year.min()+1)*365)

# %% [markdown]
# ### Check window size for moving averages

# %%
a_signal = all_locs_after_2004.loc[all_locs_after_2004.year==2004, all_locs_after_2004.columns[0]]

window_10 = 10
weights_10 = np.arange(1, window_10+1)

window_7 = 7
weights_7 = np.arange(1, window_7+1)

window_3 = 3
weights_3 = np.arange(1, window_3+1)

window_5 =5
weights_5 = np.arange(1, window_5+1)

wma_10 = a_signal.rolling(window_10).apply(lambda a_signal: np.dot(a_signal, weights_10)/weights_10.sum(), raw=True)
wma_7 = a_signal.rolling(window_7).apply(lambda a_signal: np.dot(a_signal, weights_7)/weights_7.sum(), raw=True)
wma_5 = a_signal.rolling(window_5).apply(lambda a_signal: np.dot(a_signal, weights_5)/weights_5.sum(), raw=True)
wma_3 = a_signal.rolling(window_3).apply(lambda a_signal: np.dot(a_signal, weights_3)/weights_3.sum(), raw=True)

fig, ax2 = plt.subplots(1, 1, figsize=(15, 5), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

ax2.plot(a_signal, linewidth = 2, ls = '-', label = 'noisy signal', c="dodgerblue");
# ax2.plot(wma_10, linewidth = 4, ls = '-', label = 'wma_10', c="k");
ax2.plot(wma_7, linewidth = 4, ls = '-', label = 'WMA_7', c="r");
ax2.plot(wma_5, linewidth = 2, ls = '-', label = 'WMA_5', c="k");
ax2.plot(wma_3, linewidth = 2, ls = '-', label = 'WMA_3', c="g");

ax2.tick_params(axis = 'y', which = 'major') #, labelsize = tick_FontSize) #
ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) # 
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter('%b'))

ax2.legend(loc="upper right");
ax2.grid(True);

# %%
a_signal

# %% [markdown]
# ### Smooth the dataframe by window size 5.

# %%
all_locs_smooth_after_2004 = all_locs_after_2004.copy()

window_5 = 5
weights_5 = np.arange(1, window_5+1)

for a_loc in locations:
    a_signal = all_locs_smooth_after_2004[a_loc]
    wma_5 = a_signal.rolling(window_5).apply(lambda a_signal: np.dot(a_signal, weights_5)/weights_5.sum(), raw=True)
    all_locs_smooth_after_2004[a_loc] = wma_5

all_locs_smooth_after_2004.head(10)

# %% [markdown]
# We lost some data at the beginning due to rolling window. So, we replace them here:

# %%
end = window_5-1
all_locs_smooth_after_2004.iloc[0:end, 0:len(locations)]=all_locs_smooth_after_2004.iloc[end, 0:len(locations)]

# all_locs_smooth_after_2004 = all_locs_smooth_after_2004.assign(time_xAxis=range(len(all_locs_smooth_after_2004)))
all_locs_smooth_after_2004.head(2)

# %%
years = all_locs_smooth_after_2004.year.unique()

# %%
a_year = years[9]
a_loc = locations[0]
a_year_data = all_locs_smooth_after_2004.loc[all_locs_smooth_after_2004.year==a_year, a_loc]
a_year_data_2D = np.concatenate((np.array(a_year_data).reshape(-1,1), 
                                 np.array(range(len(a_year_data))).reshape(-1,1)), 
                                 axis=1)
# ripser.ripser(all_locs_smooth_after_2004[["time_xAxis", "48.97191_-121.05145"]])['dgms']
a_dmg = ripser.ripser(a_year_data_2D)['dgms']

# plt.plot(dgm_noisy_0_x, dgm_noisy_0_y, 'g^', dgm_noisy_1_x, dgm_noisy_1_y, 'bs');
fig, axs = plt.subplots(1, 1, figsize=(10, 2.5), sharex=False, sharey=False, # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.15, 'wspace': .15});
(ax1) = axs; # ax2
ax1.grid(True); # ax2.grid(True);

ax1.plot(list(a_year_data), linewidth = 2, ls = '-', c="dodgerblue", label="brightness diff.")
ax1.set_title(f"brightness diff. (Smoothed, wma ={window_5})")
ax1.legend(loc="lower right");
ax1.set_ylabel('brightness diff.');
ax1.set_xlabel('day of year');

# %%
persim.plot_diagrams(a_dmg, show=False, title=f"rips output\n{diagram_sizes(a_dmg)}")

# %%

# %%
# The following is from https://kepler-mapper.scikit-tda.org/en/latest/started.html.
# I just changed the names data and labels to circles_data, circles_labels
# Import the class
import kmapper as km

# Some sample data
from sklearn import datasets
circles_data, circles_labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
projected_data = mapper.fit_transform(circles_data, projection=[0,1]) # X-Y axis

# Create a cover with 10 elements
cover = km.Cover(n_cubes=10)

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, circles_data, cover=cover)

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                  title="make_circles(n_samples=5000, noise=0.03, factor=0.3)");

# %%
fig, (ax1) = plt.subplots(1, 1, figsize=(6,4))
ax1.set_xlim([np.min(circles_data[:,0])-.2, np.max(circles_data[:,0])+.2]);
ax1.set_ylim([np.min(circles_data[:,1])-.2, np.max(circles_data[:,1])+.2]);
ax1.scatter(circles_data[:,0], circles_data[:,1], c=circles_labels, cmap=plt.get_cmap('winter'));
ax1.axis('off');

# %%
all_locs_smooth_after_2004_TS = np.array(all_locs_smooth_after_2004.iloc[:, :-4].T)

# %%
all_locs_smooth_after_2004_TS.shape

# %%

# %%
