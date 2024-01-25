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
import numpy as np

# %%
ideal = np.array([0, 0, 100, 0, 0])

pd1 = np.array([0, 30, 40, 30, 0])
pd2 = np.array([30, 0, 40, 0, 30])

weights = np.array([3, 2, 1, 2, 3])

# %%
# %%time
for count in np.arange(3):
    print (f"{count = }")
print ()

# %%
no_runs = 10000000

# %% [markdown]
# # Method 1

# %%
# %%time
for count in np.arange(no_runs):
    diff_ = pd1 - ideal
    diff_squared = diff_ ** 2 # elementwise square
    inner_product = np.dot(weights, diff_squared)
    distance = np.sqrt(inner_product)

# %% [markdown]
# # Method 2
#
# **element-wise product**

# %%
# %%time
for count in np.arange(no_runs):
    diff_ = pd1 - ideal
    # element-wise product 1
    diff_squared = np.multiply(diff_, diff_)
    
    # element-wise product 2
    inner_product = np.multiply(weights, diff_squared)
    distance = np.sqrt(np.sum(inner_product))

# %% [markdown]
# # Method 3
#
# **Matrix form**

# %%
weight_matrix = np.diag(weights)
weight_matrix

# %%
# %%time
for count in np.arange(no_runs):
    diff_ = pd1 - ideal
    
    # matrix product
    matrix_prod = np.dot(diff_, weight_matrix)
    matrix_prod = np.dot(matrix_prod, diff_)
    distance = np.sqrt(matrix_prod)


# %%
def weighted_euclidean(vec1, vec2, weights_):
    diff_ = vec1 - vec2
    diff_squared = diff_ ** 2 # elementwise square
    inner_product = np.dot(weights_, diff_squared)
    distance = np.sqrt(inner_product)
    return (distance.round(2))


# %%
weighted_euclidean(vec1=ideal, vec2=pd1, weights_=weights)

# %%
weighted_euclidean(vec1=ideal, vec2=pd2, weights_=weights)

# %%
