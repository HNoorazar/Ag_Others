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
import matplotlib.pyplot as plt\


import os

# %%
outdir = "/Users/hn/Documents/01_research_data/Other_people/Ehsan/"
os.makedirs(outdir, exist_ok=True)

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
file_name = outdir + f"degree7PolyModelRemoveOutlier.pdf"
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
file_name = outdir + f"trainTestMSE.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)

# %%
train_df.shape

# %%
test_df.shape

# %%
###

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=False, gridspec_kw={"hspace": 0.02, "wspace": 0.1})
axes.grid(axis="y", which="both");


x = np.linspace(-10, 10, 400)
y = x ** 2

axes.plot(x, y, label = 'true model', lw=3);

axes.axhline(y=55, label="model 1", color="y", lw=3);
axes.axhline(y=65, label="model 2", color="r", lw=3);
axes.axhline(y=70, label="model 3", color="g", lw=3);
axes.axhline(y=58, label="model 4", color="orange", lw=3);

axes.legend(loc="best");
axes.set_xlim(3, 10);

plt.title('models based on different datasets');
file_name = outdir + f"crudeModelVariance.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)

# %%

# %%
x_skewed = np.concatenate((np.linspace(-10, -1, 50), np.linspace(-1, 10, 10))) 
x_skewed.sort()
x_skewed = x_skewed+2
y_skewed = (x_skewed**2)/10


fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=False, gridspec_kw={"hspace": 0.02, "wspace": 0.1})
axes.grid(axis="y", which="both");
axes.plot(x_skewed, y_skewed, label = 'train error', lw=3);
axes.legend(loc="best");

# %%
x_skewed = np.concatenate((np.linspace(-10, -1, 50), np.linspace(-1, 10, 10))) 
x_skewed.sort()
x_skewed = x_skewed + 3
y_skewed = (x_skewed**2)


fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=False, gridspec_kw={"hspace": 0.02, "wspace": 0.1})
axes.grid(axis="y", which="both");
axes.plot(x_skewed, y_skewed, label = 'train error', lw=3);
axes.legend(loc="best");

# %%

# %%

# %%
# Original x_skewed and y_skewed
x_skewed = np.concatenate((np.linspace(-10, -1, 50), np.linspace(-1, 10, 10))) 
x_skewed.sort()
x_skewed = x_skewed + 2
y_skewed = (x_skewed**2)/10

# x and y for the second plot
x = np.linspace(0, 10, 400)
y = 1 / x**(1/3)

x = x[1:]; y = y[1:]

# Shifting and scaling y_skewed so it's above the y plot
y_skewed_shifted = y_skewed + np.max(y) + 0.2  # Add a vertical offset

# Plotting the original x and y, and the shifted x_skewed and y_skewed
plt.figure(figsize=(8, 6))

# Plotting y vs x
plt.plot(x, y, label='y = 1/x^(1/3)', color='b', linewidth=2)

# Plotting the shifted y_skewed vs x_skewed
plt.plot(x_skewed, y_skewed_shifted, label='Shifted y_skewed', color='r', linestyle='--', linewidth=2)

# Labels and Title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Shifting and Scaling x_skewed, y_skewed')

# Show legend
plt.legend()

# Show grid and plot
plt.grid(True)
plt.show()

# %%

# %%

# %%
# X-axis: model complexity
x = np.linspace(0, 10, 300)

# Simulated training error: decreases monotonically
train_error = np.exp(-0.3 * x)

# Simulated generalization (true) error: U-shape + noise
true_error = 0.1 + 0.05 * (x - 5)**2 + 0.05 * np.sin(2 * x)

# Simulated test error: true error + noise
test_error = true_error + 0.03 * np.random.randn(len(x))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, train_error, label="Training Error", linewidth=2)
plt.plot(x, test_error, label="Test Error", linewidth=2)
plt.plot(x, true_error, label="Generalization (True) Error", linestyle='--', linewidth=2)

# Marking w' and w''
w1 = 3.5
w2 = 7.5
plt.axvline(w1, color='gray', linestyle=':', ymax=0.65)
plt.axvline(w2, color='gray', linestyle=':', ymax=0.85)
plt.text(w1, 0.4, "w'", ha='center', fontsize=12)
plt.text(w2, 0.6, "w''", ha='center', fontsize=12)

# Labels and styling
plt.xlabel("Model Complexity")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.title("Training, Test, and Generalization Error vs. Model Complexity")

plt.tight_layout()
plt.show()

# %%
# X-axis: model complexity
x = np.linspace(0, 10, 300)

# Simulated training error: decreases monotonically
train_error = np.exp(-0.3 * x)

# Simulated generalization (true) error: U-shape + oscillation, shifted upward
true_error = 0.1 + 0.05 * (x - 5)**2 + 0.05 * np.sin(2 * x) + 0.3  # +0.3 ensures it's above training error
test_error = true_error + 0.03 * np.random.randn(len(x))

# Plotting
plt.figure(figsize=(5, 5))
plt.plot(x, train_error, label="training Error", linewidth=2)
plt.plot(x, test_error, label="test Error", linewidth=1)
plt.plot(x, true_error, label="generalization (true) Error", linestyle='--', linewidth=2)


# Marking w' and w''
w1, w2 = 5, 7.5
plt.axvline(w1, color='gray', linestyle=':')
plt.axvline(w2, color='gray', linestyle=':')
plt.text(w1-.4, 1.5, r"$\hat \beta_1$", ha='center', fontsize=12)
plt.text(w2-.4, 1.5, r"$\hat \beta_2$", ha='center', fontsize=12)

# Labels and styling
plt.xlabel("model flexibility"); plt.ylabel("error");
plt.xticks([]); plt.yticks([]);

plt.legend(loc="lower left")
plt.grid(True)
plt.title("training and generalization error vs. model complexity")

plt.tight_layout()

file_name = outdir + f"GeneralizationOverFit.pdf"
plt.savefig(file_name, bbox_inches="tight", dpi=300)

# %%
