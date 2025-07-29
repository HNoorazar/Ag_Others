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
outdir = "/Users/hn/Documents/01_research_data/Other_people/Ehsan/"

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
x = np.random.normal(size=vec_size)
y = x + np.random.normal(loc=0, scale=1, size=vec_size)
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
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")

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

# %% [markdown]
# # Remove one outlier and see its effect on the model

# %%
Y_train_outlier = Y_train.copy()
Y_train_outlier[0] = 20
Y_train_outlier

# %%
model_outlier = sm.OLS(Y_train_outlier, X_train);
model_result_outlier = model_outlier.fit();
y_pred_outlier = model_result_outlier.predict(poly_df[list(X_train.columns)])
# model_result_outlier.summary()

# %%

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
(ax0, ax1, ax2) = axes
ax0.grid(axis="y", which="both"); ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")

ax0.scatter(train_df["x"], train_df["y"], color="dodgerblue", label = "data", s=50, marker='s');
ax0.scatter(train_df["x"], Y_train_outlier, color="red", label = "data w. outlier", s=30);
ax0.plot(poly_df["x"], poly_df["y_pred"], color="dodgerblue", label="model");
ax0.plot(poly_df["x"], y_pred_outlier, color="red", label="model w. outlier");
ax0.legend(loc="best");
ax0.axvline(x=X_train['x'][0], color='k', linestyle='--')
##################################
##################################
##################################
ax1.scatter(train_df["x"], train_df["y"], color="dodgerblue", label = "data", s=50, marker='s');
ax1.scatter(train_df["x"], Y_train_outlier, color="red", label = "data w. outlier", s=30);
ax1.plot(poly_df["x"], poly_df["y_pred"], color="dodgerblue", label="model");
ax1.plot(poly_df["x"], y_pred_outlier, color="red", label="model w. outlier");
ax1.legend(loc="best");
ax1.axvline(x=X_train['x'][0], color='k', linestyle='--')
ax1.set_ylim(-200, 200);
legend = ax1.get_legend()
if legend is not None:
    legend.remove()
##################################
##################################
##################################
ax2.scatter(train_df["x"], train_df["y"], color="dodgerblue", label = "data", s=50, marker='s');
ax2.scatter(train_df["x"], Y_train_outlier, color="red", label = "data w. outlier", s=30);
ax2.plot(poly_df["x"], poly_df["y_pred"], color="dodgerblue", label="model");
ax2.plot(poly_df["x"], y_pred_outlier, color="red", label="model w. outlier");
ax2.axvline(x=X_train['x'][0], color='k', linestyle='--', label='My Vertical Line')
ax2.set_ylim(-2, 25);
# Add arrow to the vertical line
ax2.annotate('',                    # No text
             xy=(X_train['x'][0], 15),    # Arrow head
             xytext=(X_train['x'][0], 3), # Arrow tail
             arrowprops=dict(facecolor='k', shrink=0.05, width=2, headwidth=8))

legend = ax2.get_legend()
if legend is not None:
    legend.remove()
    
fig.suptitle(f"effect of removing an outlier on a highly flexible model", y=0.95, 
             fontdict={"family": "serif"}, fontsize=15)
file_name = outdir + f"degree7PolyModelRemoveOutlier.png"
plt.savefig(file_name, bbox_inches="tight", dpi=300)

# %%

# %%
model_result_outlier.params

# %%
model_result.params

# %%
train_df.head(3)

# %%
train_df.iloc[:, 1:]

# %%
model_d1 = sm.OLS(Y_train, train_df.iloc[:, 1:3]);
model_result_d1 = model_d1.fit();
model_result_d1.params

# %%
model_d7 = sm.OLS(Y_train, train_df.iloc[:, 1:9]);
model_result_d7 = model_d7.fit();
model_result_d7.params

# %%
model_d12 = sm.OLS(Y_train, train_df.iloc[:, 1:13]);
model_result_d12 = model_d12.fit();
model_result_d12.params

# %%
y_pred_d12 = model_result_d12.predict(train_df.iloc[:, 1:])

# %%
from sklearn.metrics import mean_squared_error

# %%
## record test and train errors for polynomials
train_mse_list = []
test_mse_list = []
for degree in np.arange(3, 14):
    model = sm.OLS(Y_train, train_df.iloc[:, 1:degree]);
    model_result = model.fit();
    train_yhat = model_result.predict(train_df.iloc[:, 1:degree])
    test_yhat = model_result.predict(test_df.iloc[:, 1:degree])
    
    train_mse = mean_squared_error(train_df["y"], train_yhat)
    test_mse = mean_squared_error(test_df["y"], test_yhat)
    
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
    

# %%
train_mse_list

# %%
test_mse_list

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, gridspec_kw={"hspace": 0.02, "wspace": 0.1})
(ax1, ax2) = axes
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")

ax1.plot(np.arange(len(train_mse_list)), train_mse_list, label = "train MSE");
ax1.plot(np.arange(len(test_mse_list)), test_mse_list, label = "train MSE");
ax1.legend(loc="best");
###############################
###############################
###############################
ax2.plot(np.arange(len(train_mse_list)), train_mse_list, label = "train MSE");
ax2.plot(np.arange(len(test_mse_list)), test_mse_list, label = "train MSE");
ax2.legend(loc="best");
ax2.set_ylim(0, 10);


ax1.set_xlabel("model flexibility (polynomial degree)");
ax2.set_xlabel("model flexibility (polynomial degree)");
ax1.set_ylabel("mean squared error");

fig.suptitle(f"train and test MSE as model flexibility increases", y=0.95, 
             fontdict={"family": "serif"}, fontsize=15)
file_name = outdir + f"trainTestMSE.png"
plt.savefig(file_name, bbox_inches="tight", dpi=300)

# %%
train_df.shape

# %%
test_df.shape

# %%
