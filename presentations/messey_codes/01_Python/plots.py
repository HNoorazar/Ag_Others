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
import pandas as pd
import matplotlib.pyplot as plt

# %%
y_test_cs= pd.read_excel('y_test_cs.xlsx')
y_test_pred_cs= pd.read_excel('y_test_pred_cs.xlsx')

y_test_ridge= pd.read_excel('y_test_ridge.xlsx')
y_test_pred_ridge= pd.read_excel('y_test_pred_ridge.xlsx')

y_test_rf= pd.read_excel('y_test_rf.xlsx')
y_test_pred_rf= pd.read_excel('y_test_pred_rf.xlsx')

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharex=True, sharey=True)

# Scatter plot for CS Model
axes[0].scatter(y_test_cs, y_test_pred_cs, alpha=0.7, color='blue')
axes[0].plot(y_test_cs, y_test_cs, color='red', linestyle='--')
axes[0].set_title('CropSyst')
axes[0].legend()
# Add text box
axes[0].text(0.05, 0.95, 'Cor. Coef.: 0.38', transform=axes[0].transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# Scatter plot for Ridge Model
axes[1].scatter(y_test_ridge, y_test_pred_ridge, alpha=0.7, color='green')
axes[1].plot(y_test_ridge, y_test_ridge, color='red', linestyle='--')
axes[1].set_title('Ridge Regression')
axes[1].legend()
# Add text box
axes[1].text(0.05, 0.95, 'Cor. Coef.: 0.65', transform=axes[1].transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# Scatter plot for RF Model
axes[2].scatter(y_test_rf, y_test_pred_rf, alpha=0.7, color='purple')
axes[2].plot(y_test_rf, y_test_rf, color='red', linestyle='--')
axes[2].set_title('Random Forest')
axes[2].legend()
# Add text box
axes[2].text(0.05, 0.95, 'Cor. Coef.: 0.76', transform=axes[2].transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# Set common x and y labels
fig.text(0.5, 0.02, 'True Values', ha='center', fontsize=12)  # x-axis title
fig.text(0.02, 0.5, 'Predicted Values', va='center', rotation='vertical', fontsize=12)  # y-axis title

plt.tight_layout(rect=[0.03, 0.03, 1, 1]) 



# %%
y_test_ridge= pd.read_excel('y_test_ridge_tw.xlsx')
y_test_pred_ridge= pd.read_excel('y_test_pred_ridge_tw.xlsx')

y_test_rf= pd.read_excel('y_test_rf_tw.xlsx')
y_test_pred_rf= pd.read_excel('y_test_pred_rf_tw.xlsx')

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 3), sharex=True, sharey=True)


# Scatter plot for RF Model
axes[1].scatter(y_test_rf, y_test_pred_rf, alpha=0.7, color='purple')
axes[1].plot(y_test_rf, y_test_rf, color='red', linestyle='--')
axes[1].set_title('Random Forest')
axes[1].legend()
# Add text box
axes[1].text(0.05, 0.95, 'Cor. Coef.: 0.16', transform=axes[1].transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# Set common x and y labels
fig.text(0.5, 0.02, 'True Values', ha='center', fontsize=12)  # x-axis title
fig.text(0.02, 0.5, 'Predicted Values', va='center', rotation='vertical', fontsize=12)  # y-axis title

plt.tight_layout(rect=[0.03, 0.03, 1, 1]) 
plt.savefig('hd.png', dpi = 300)

# %%
import matplotlib.pyplot as plt

# Create a single figure
fig, ax = plt.subplots(figsize=(6, 6))

# Scatter plot for Random Forest Model
ax.scatter(y_test_rf, y_test_pred_rf, alpha=0.7, color='black')
ax.plot(y_test_rf, y_test_rf, color='red', linestyle='--')
#ax.set_title('Random Forest', fontsize=14)


# Add text box with Correlation Coefficient and MAE
ax.text(0.05, 0.95, 'Cor. Coef.: 0.36\nMAE (%): 3.03', transform=ax.transAxes,
        fontsize=20, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

ax.tick_params(axis='both', labelsize=20) 
#ax.tick_params(axis='y', labelsize=20)
# Set x and y labels
ax.set_xlabel('True Values', fontsize=20)
ax.set_ylabel('Predicted Values', fontsize=20)

# Set axis limits to start from zero
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Adjust layout and save
plt.tight_layout()
plt.savefig('tw.png', dpi=300)
plt.show()




# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
y_test_rf = pd.read_excel('y_test_rf_tw.xlsx')
y_test_pred_rf = pd.read_excel('y_test_pred_rf_tw.xlsx')

# Scatter plot
plt.scatter(y_test_rf, y_test_pred_rf, alpha=0.7, color='black')

# Best fit line constrained to start at (0,0)
coefficients_constrained = np.polyfit(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten(), 1)
best_fit_line_constrained = coefficients_constrained[1] + coefficients_constrained[0] * y_test_rf.values.flatten()
plt.plot(y_test_rf, best_fit_line_constrained, color='red')

# 1:1 line
x = np.linspace(0, max(y_test_rf.values.flatten()), 100)
plt.plot(x, x, color='blue', linestyle='--')

plt.tick_params(axis='both', labelsize=20)

# Add text box with Correlation Coefficient and MAE
corr_coef = np.corrcoef(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten())[0, 1]
mae = np.mean(np.abs((y_test_pred_rf.values.flatten() - y_test_rf.values.flatten()) / y_test_rf.values.flatten())) * 100

plt.text(
    0.05, 0.95, f'Cor. Coef.: {corr_coef:.2f}\nBias: {mae:.2f}', 
    transform=plt.gca().transAxes, fontsize=20, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.5)
)

# Plot settings
plt.xlabel("Actual Test Weight (lb/bu)",fontsize=20)
plt.ylabel("Predicted  Test Weight (lb/bu)",fontsize=20)
plt.grid(True)
plt.savefig('tw.png', dpi=300)
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
y_test_rf = pd.read_excel('y_test_rf_gy.xlsx')
y_test_pred_rf = pd.read_excel('y_test_pred_rf_gy.xlsx')

# Scatter plot
plt.scatter(y_test_rf, y_test_pred_rf, alpha=0.7, color='black')

# Best fit line constrained to start at (0,0)
coefficients_constrained = np.polyfit(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten(), 1)
best_fit_line_constrained = coefficients_constrained[1] + coefficients_constrained[0] * y_test_rf.values.flatten()
plt.plot(y_test_rf, best_fit_line_constrained, color='red')
plt.tick_params(axis='both', labelsize=20)
# 1:1 line
x = np.linspace(0, max(y_test_rf.values.flatten()), 100)
plt.plot(x, x, color='blue', linestyle='--')

# Add text box with Correlation Coefficient and MAE
corr_coef = np.corrcoef(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten())[0, 1]
mae = np.mean(np.abs((y_test_pred_rf.values.flatten() - y_test_rf.values.flatten()) / y_test_rf.values.flatten())) * 100

plt.text(
    0.05, 0.95, f'Cor. Coef.: {corr_coef:.2f}\nBias: {mae:.2f}',
    transform=plt.gca().transAxes, fontsize=20, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.5)
)

# Plot settings
plt.xlabel("Actual Grain Yield (bu/A)",fontsize=20)
plt.ylabel("Predicted  Grain Yield (bu/A)",fontsize=20)
plt.grid(True)
plt.savefig('gy.png', dpi=300)
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
y_test_rf = pd.read_excel('y_test_rf_hd.xlsx')
y_test_pred_rf = pd.read_excel('y_test_pred_rf_hd.xlsx')

# Scatter plot
plt.scatter(y_test_rf, y_test_pred_rf, alpha=0.7, color='black')

# Best fit line constrained to start at (0,0)
coefficients_constrained = np.polyfit(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten(), 1)
best_fit_line_constrained = coefficients_constrained[1] + coefficients_constrained[0] * y_test_rf.values.flatten()
plt.plot(y_test_rf, best_fit_line_constrained, color='red')

# 1:1 line
x = np.linspace(0, max(y_test_rf.values.flatten()), 100)
plt.plot(x, x, color='blue', linestyle='--')
plt.tick_params(axis='both', labelsize=20)
# Add text box with Correlation Coefficient and MAE
corr_coef = np.corrcoef(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten())[0, 1]
mae = np.mean(np.abs((y_test_pred_rf.values.flatten() - y_test_rf.values.flatten()) / y_test_rf.values.flatten())) * 100

plt.text(
    0.05, 0.95, f'Cor. Coef.: {corr_coef:.2f}\nBias: {mae:.2f}',
    transform=plt.gca().transAxes, fontsize=20, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.5)
)

# Plot settings
plt.xlabel("Actual Heading Date (day)",fontsize=20)
plt.ylabel("Predicted  Heading Date (day)",fontsize=20)
plt.grid(True)
plt.savefig('hd.png', dpi=300)
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
y_test_rf = pd.read_excel('y_test_rf_ph.xlsx')
y_test_pred_rf = pd.read_excel('y_test_pred_rf_ph.xlsx')

# Scatter plot
plt.scatter(y_test_rf, y_test_pred_rf, alpha=0.7, color='black')

# Best fit line constrained to start at (0,0)
coefficients_constrained = np.polyfit(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten(), 1)
best_fit_line_constrained = coefficients_constrained[1] + coefficients_constrained[0] * y_test_rf.values.flatten()
plt.plot(y_test_rf, best_fit_line_constrained, color='red')
plt.tick_params(axis='both', labelsize=20)
# 1:1 line
x = np.linspace(0, max(y_test_rf.values.flatten()), 100)
plt.plot(x, x, color='blue', linestyle='--')

# Add text box with Correlation Coefficient and MAE
corr_coef = np.corrcoef(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten())[0, 1]
mae = np.mean(np.abs((y_test_pred_rf.values.flatten() - y_test_rf.values.flatten()) / y_test_rf.values.flatten())) * 100

plt.text(
    0.05, 0.95, f'Cor. Coef.: {corr_coef:.2f}\nBias: {mae:.2f}',
    transform=plt.gca().transAxes, fontsize=20, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.5)
)

# Plot settings
plt.xlabel("Actual Plant Height (inch)",fontsize=20)
plt.ylabel("Predicted  Plant Height (inch)",fontsize=20)
plt.grid(True)
plt.savefig('ph.png', dpi=300)
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
y_test_rf = pd.read_excel('y_test_rf_pc.xlsx')
y_test_pred_rf = pd.read_excel('y_test_pred_rf_pc.xlsx')

# Scatter plot
plt.scatter(y_test_rf, y_test_pred_rf, alpha=0.7, color='black')

# Best fit line constrained to start at (0,0)
coefficients_constrained = np.polyfit(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten(), 1)
best_fit_line_constrained = coefficients_constrained[1] + coefficients_constrained[0] * y_test_rf.values.flatten()
plt.plot(y_test_rf, best_fit_line_constrained, color='red')
plt.tick_params(axis='both', labelsize=20)
# 1:1 line
x = np.linspace(0, max(y_test_rf.values.flatten()), 100)
plt.plot(x, x, color='blue', linestyle='--')

# Add text box with Correlation Coefficient and MAE
corr_coef = np.corrcoef(y_test_rf.values.flatten(), y_test_pred_rf.values.flatten())[0, 1]
mae = np.mean(np.abs((y_test_pred_rf.values.flatten() - y_test_rf.values.flatten()) / y_test_rf.values.flatten())) * 100

plt.text(
    0.05, 0.95, f'Cor. Coef.: {corr_coef:.2f}\nBias: {mae:.2f}',
    transform=plt.gca().transAxes, fontsize=20, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.5)
)

# Plot settings
plt.xlabel("Actual Prtein Content (%)",fontsize=20)
plt.ylabel("Predicted  Prtein Content (%)",fontsize=20)
plt.grid(True)
plt.savefig('pc.png', dpi=300)
plt.show()

