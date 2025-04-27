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

# %%
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# ### Directories

# %%
wheat_database = "/Users/hn/Documents/01_research_data/Ehsan/wheat/"
separate_varieties_dir = wheat_database + "varieties/"

# %%
# perform pairwise correlation analysis
df_env = pd.read_csv('df_22805.csv')
df = df_env.drop(columns=['location', 'year', 'Grain_yield', 'Heading_date', 'Height','Protein'])

"""
###########################
###   Encode soil type  ###
###########################
# Assuming 'soil_type' is the column in your DataFrame
label_encoder = LabelEncoder()
df['soil_type_encoded'] = label_encoder.fit_transform(df['soil_type'])

# Optional: Check the mapping
soil_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding Mapping:", soil_type_mapping)
df = df.drop('soil_type', axis = 1)
df.rename(columns={"soil_type_encoded": "Soil_type"}, inplace=True)

# Impute missing values (column-wise)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Perform pairwise correlation analysis
corr_matrix = df_imputed.corr()
"""

# %%

# %%
# Compute the correlation matrix
corr_matrix = df.corr().abs()  # Absolute correlation values

# Unstack the matrix to get pairs
corr_pairs = corr_matrix.unstack()

# Remove duplicate pairs and self-correlations
filtered_pairs = corr_pairs[(corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1))]
unique_pairs = filtered_pairs.loc[filtered_pairs.index.map(frozenset).duplicated(keep='first') == False]

# Sort the pairs by correlation value
sorted_pairs = unique_pairs.sort_values(ascending=False)

# Filter for top correlations (adjust threshold as needed)
top_pairs = sorted_pairs[sorted_pairs > 0.5]  # Example threshold: > 0.5
top_pairs

# %%
import pandas as pd

# Define trait mappings
traits_dict = {
    'Grain_yield': 'gy',
    'Height': 'ph',
    'Protein': 'pc',
    'Test_weight': 'tw'
}

# Read the dataset once
df_env = pd.read_csv('df_22805.csv')

# Columns to drop that should not be included in the correlation analysis
drop_columns = ['location', 'year']  # Removed 'Heading_date' since it's not in the dataset

# Perform analysis for each trait
for trait, trait_acr in traits_dict.items():
    df = df_env.drop(columns=drop_columns)  # Drop unwanted columns

    # Remove the other three traits (keep only the target trait)
    traits_to_remove = [t for t in traits_dict if t != trait]
    df = df.drop(columns=traits_to_remove)

    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Identify pairs of highly correlated variables (correlation > 0.6)
    high_corr_pairs = []
    for i, col1 in enumerate(df.columns[:-1]):  # Exclude the target column
        for col2 in df.columns[i+1:-1]:
            if abs(correlation_matrix.loc[col1, col2]) > 0.6:
                high_corr_pairs.append((col1, col2))

    # Decide which variable to remove based on correlation with target trait
    removed_features = set()
    for col1, col2 in high_corr_pairs:
        corr1 = abs(correlation_matrix.loc[col1, trait])
        corr2 = abs(correlation_matrix.loc[col2, trait])
        if corr1 < corr2:
            removed_features.add(col1)
        else:
            removed_features.add(col2)

    # Filter the DataFrame to remove the highly correlated variables
    filtered_df = df.drop(columns=list(removed_features))
    
    # Add year location back to the filtered DataFrame
    filtered_df = pd.concat([df_env[drop_columns], filtered_df], axis=1)
    
    # Add a new column named environment by combining location and year
    filtered_df['environment'] = filtered_df['location'].astype(str) + '_' + filtered_df['year'].astype(str)
    # Save the filtered dataset
    output_filename = f'{trait_acr}_33125.csv'
    filtered_df.to_csv(output_filename, index=False)

#     print(f"Removed features for {trait}: {removed_features}")


# %%
#----------------------------#
#        HEADING DATE        #
#----------------------------#
# perform pairwise correlation analysis
df_env = pd.read_csv('df_22805_hd.csv')
df = df_env.drop(columns=['location', 'year'])

# Compute the correlation matrix
correlation_matrix = df.corr()

# Identify pairs of highly correlated variables (correlation > 0.6)
high_corr_pairs = []
for i, col1 in enumerate(df.columns[:-1]):  # Exclude the target column
    for col2 in df.columns[i+1:-1]:
        if abs(correlation_matrix.loc[col1, col2]) > 0.7:
            high_corr_pairs.append((col1, col2))

# Decide which variable to remove based on correlation with target
removed_features = set()
for col1, col2 in high_corr_pairs:
    corr1 = abs(correlation_matrix.loc[col1, 'Heading_date'])
    corr2 = abs(correlation_matrix.loc[col2, 'Heading_date'])
    if corr1 < corr2:
        removed_features.add(col1)
    else:
        removed_features.add(col2)

# Filter the DataFrame to remove the highly correlated variables
filtered_df = df.drop(columns=list(removed_features))
# Add year location back to the filtered DataFrame
filtered_df = pd.concat([df_env[drop_columns], filtered_df], axis=1)
# Add a new column named environment by combining location and year
filtered_df['environment'] = filtered_df['location'].astype(str) + '_' + filtered_df['year'].astype(str)
filtered_df.to_csv('hd_33125.csv', index  = False)

removed_features

# %%
max_value = correlation_matrix['Longitude'].nlargest(10)
max_value

# %%
import pandas as pd
import numpy as np
import plotly.express as px

# Generate a sample dataset with 200 features and 500 rows
data = df_imputed

# Calculate the correlation matrix
corr_matrix = data.corr()

# Keep only the upper triangle of the correlation matrix
mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
corr_matrix = corr_matrix.where(mask)

# Interactive heatmap with Plotly
fig = px.imshow(
    corr_matrix,
    color_continuous_scale='rainbow',  # A valid Plotly colorscale
    title="Interactive Correlation Matrix Heatmap (Upper Triangle)",
    labels={'x': "Features", 'y': "Features", 'color': "Correlation"}
)
fig.update_layout(width=1000, height=1000)
fig.show()
fig.write_html("correlation_heatmap_hd.html")
