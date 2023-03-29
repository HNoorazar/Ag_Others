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
### Neighbor Joining from Biology

# %%
from skbio import DistanceMatrix
from skbio.tree import nj

data = [[0,  5,  9,  9,  8],
         [5,  0, 10, 10,  9],
         [9, 10,  0,  8,  7],
         [9, 10,  8,  0,  3],
         [8,  9,  7,  3,  0]]

# data = pd.DataFrame(data, columns=ids, index=ids)

ids = list('abcde')
dm = DistanceMatrix(data, ids)

tree = nj(dm)
print(tree.ascii_art())
print ("--------------------------------------------------------")
newick_str = nj(dm, result_constructor=str)
print(newick_str)




# %%
dm = DistanceMatrix(data)
dm

# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np

mat = np.array([[0, 3, 0.1], [3, 0, 2], [0.1, 2, 0]])
dists = squareform(mat)
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels=["0", "1", "2"])
plt.title("test")
plt.show()

print (f"{linkage_matrix}")
print ("------------------------------")
print (f"{mat}")

# %%
