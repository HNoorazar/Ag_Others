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

# %% [markdown]
# ## Overfitting or not. That's the question

# %%
import pandas as pd
import numpy as np

import random

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt

# %%
random.seed(10)
print(random.random())


# %%
np.random.seed(10)
np.random.normal(loc=0.0, scale=1.0, size=10)

# %% [markdown]
# ### First create `x` and then add random noise to each `x` to get `y`.
#
# So, some random points around the $y=x$ line

# %%
np.random.seed(6)
vec_size = 10
x = np.random.normal(size =vec_size)
y = x + np.random.normal(loc=0, scale =1, size =vec_size)
# np. corrcoef (x, y)

# %%
x_y_line = np.arange(-4, 5)

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=False)
axes.grid(axis="y", which="both");
axes.plot(x_y_line, x_y_line, label = "$y=x$")

axes.scatter(x, y);

# %% [markdown]
# ### add powers of `x` to the dataframe for polynomial regression

# %%
df = pd.DataFrame(columns=["y", "x"], data= np.column_stack((y, x)) )
for a in np.arange(2, 12):
    df["x" + str(a)] = x**a

df.head(2)

# %%
df = sm.add_constant(df)
df.head(3)

# %%

# %%
# reorder columns!
a = list(df.columns)
a.remove("y")
df = df[['y'] + a]
df.head(3)

# %%
# Everything has been random. 
# so, no need to "shuffle"
train_df = df.loc[0:7].copy()
test_df =  df.loc[8:].copy()

# %%
X_train = train_df.iloc[:, 1:9]
Y_train = train_df.iloc[:, 0]
X_train.head(2)

# %%
import warnings
warnings.filterwarnings('ignore') 

model = sm.OLS(Y_train, X_train);
model_result = model.fit();
model_result.summary()

# %%
X_test = test_df.iloc[:, 1:9]
Y_test = test_df.iloc[:, 0]

# %%

# %%
#### polynomial creation to show fitted model
x = np.linspace(start = np.floor(df["x"].min()), stop = np.ceil(df["x"].max()), num=100)
poly_df = pd.DataFrame(columns=["x"], data=x)
for a in np.arange(2, 12):
    poly_df["x" + str(a)] = x**a

poly_df = sm.add_constant(poly_df)
poly_df["y_pred"] = model_result.predict(poly_df[list(X_train.columns)])

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")

ax1.plot(x_y_line, x_y_line, label = "$y=x$")

ax1.scatter(train_df["x"], train_df["y"], color="dodgerblue", label = "train GT", s=74);
ax1.scatter(test_df["x"],  test_df["y"], color="red", label = "test GT");


ax1.scatter(train_df["x"], model_result.predict(X_train), s = 12, color="k", label="train predictions")
ax1.scatter(test_df["x"], model_result.predict(X_test), s = 72, color="c", label="test predictions")
ax1.plot(poly_df["x"], poly_df["y_pred"], label="fitted model");
ax1.legend(loc="best");
ax1.set_ylim(-200, 200);
###################################

ax2.plot(x_y_line, x_y_line, label = "$y=x$")

ax2.scatter(train_df["x"], train_df["y"], color="dodgerblue", label = "train GT", s=74);
ax2.scatter(test_df["x"],  test_df["y"], color="red", label = "test GT");


ax2.scatter(train_df["x"], model_result.predict(X_train), s = 12, color="k", label="train predictions")
ax2.scatter(test_df["x"], model_result.predict(X_test), s = 72, color="c", label="test predictions")
ax2.plot(poly_df["x"], poly_df["y_pred"], label="fitted model");
ax2.title.set_text('same as left plot; just zoomed in')

ax2.legend(loc="best");
ax2.set_ylim(-2, 10);

# %%
from scipy.linalg import norm

train_error = norm(model_result.predict(X_train) - Y_train)
print (f"{train_error = }")
print ()


test_error = norm(model_result.predict(X_test) - Y_test)
print (f"{test_error = }")

# %%

# %%
