#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:23:24 2024

@author: geet
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Function to calculate physicochemical properties from SMILES
def calculate_physicochemical_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([np.nan] * 5)  # Return NaN if SMILES is invalid
    mw = Descriptors.MolWt(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    return pd.Series([mw, h_donors, h_acceptors, tpsa, rot_bonds])

# Load datasets
raw_data = pd.read_csv('input.csv')  # Contains molecular weights
compounds_data = pd.read_csv('cluster.csv') # Contains compounds with cluster labels

# Ensure ChEMBLID and Smiles are treated as strings, and Cluster is numeric
raw_data['ChEMBLID'] = raw_data['Molecule ChEMBL ID'].astype(str)
raw_data = raw_data[['ChEMBLID', 'Molecular Weight']]

compounds_data['ChEMBLID'] = compounds_data['ChEMBLID'].astype(str)
compounds_data['Smiles'] = compounds_data['Smiles'].astype(str)

# Extract needed compounds from compounds_data
needed_compounds = compounds_data[['ChEMBLID', 'Smiles', 'IC50', 'Cluster']]

# Sort and drop duplicates without using inplace=True
needed_compounds = needed_compounds.sort_values('IC50', ascending=False)
needed_compounds = needed_compounds.drop_duplicates(subset='ChEMBLID', keep='first')

# Merge the raw data with needed compounds on ChEMBLID
merged_data = pd.merge(needed_compounds, raw_data, on='ChEMBLID', how='left')

# Calculate missing physicochemical properties from SMILES
physicochemical_properties = merged_data['Smiles'].apply(calculate_physicochemical_properties)
physicochemical_properties.columns = ['MW_Calculated', 'H_Donors', 'H_Acceptors', 'TPSA', 'Rotatable_Bonds']

# Add MW from raw data, fill in missing MW if needed without inplace=True
merged_data['MW'] = physicochemical_properties['MW_Calculated']
merged_data['MW'] = merged_data['MW'].fillna(merged_data['Molecular Weight'])

# Combine physicochemical properties into the dataset
full_properties = pd.concat([merged_data[['ChEMBLID', 'Cluster', 'IC50_nM']], physicochemical_properties.drop(columns='MW_Calculated')], axis=1)

# Convert all columns to numeric types, forcing errors to NaN
for col in full_properties.columns:
    if col not in ['ChEMBLID', 'Cluster']:
        full_properties[col] = pd.to_numeric(full_properties[col], errors='coerce')

# Handle NaN values by filling them with the mean of each column
full_properties.fillna(full_properties.mean(numeric_only=True), inplace=True)

# Remove duplicates with the same ChEMBLID, keeping the one with the highest IC50
unique_properties = full_properties.sort_values('IC50', ascending=False).drop_duplicates(subset='ChEMBLID', keep='first')

# Save the final unique data to CSV
unique_properties.to_csv('complete_physicochemical_properties.csv', index=False)

# Prepare data for Euclidean distance calculation and clustering
clusters = unique_properties['Cluster'].unique()
clusters.sort()
cluster_dict = {k: f'Cluster {k}' for k in clusters}
unique_properties['Cluster'] = unique_properties['Cluster'].map(cluster_dict)

# Remove clusters with fewer than 10 compounds
cluster_sizes = unique_properties['Cluster'].value_counts()
valid_clusters = cluster_sizes[cluster_sizes >= 10].index
filtered_properties = unique_properties[unique_properties['Cluster'].isin(valid_clusters)]

# Group by cluster and calculate the mean of physicochemical properties per cluster
cluster_means = filtered_properties.groupby('Cluster').mean(numeric_only=True)

# Ensure all clusters are present
print("Clusters in filtered_properties:", filtered_properties['Cluster'].unique())
print("Clusters in cluster_means:", cluster_means.index)

# Compute pairwise Euclidean distances between clusters
distance_matrix = pairwise_distances(cluster_means, metric='euclidean')

# Calculate intra-cluster correlations (mean distances within each cluster)
intra_cluster_distances = np.zeros(len(cluster_means))
for i, cluster in enumerate(cluster_means.index):
    cluster_indices = filtered_properties[filtered_properties['Cluster'] == cluster].index
    if len(cluster_indices) > 1:
        cluster_distances = pairwise_distances(filtered_properties.loc[cluster_indices].drop(columns=['ChEMBLID', 'Cluster', 'IC50']), metric='euclidean')
        intra_cluster_distances[i] = np.nanmean(cluster_distances)
    else:
        intra_cluster_distances[i] = 0  # Assign 0 if only one compound in the cluster

# Add intra-cluster distances to the diagonal of the distance matrix
for i, cluster in enumerate(cluster_means.index):
    distance_matrix[i, i] = intra_cluster_distances[i]

# Create a DataFrame for the distance matrix with appropriate labels
distance_df = pd.DataFrame(distance_matrix, index=cluster_means.index, columns=cluster_means.index)

# Create a mask for the upper triangle, excluding the diagonal
mask = np.triu(np.ones_like(distance_df, dtype=bool), k=1)

# Generate the heatmap with masking and custom color scheme
plt.figure(figsize=(12, 10))
cmap = sns.diverging_palette(220, 20, as_cmap=True)  # Use a custom diverging palette for dark blue to dark red

ax = sns.heatmap(distance_df, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={"label": "Euclidean Distance"}, square=True)

# Rotate axis labels for better readability
plt.xticks(rotation=90, ha='center')
plt.yticks(rotation=0)

plt.title("Intra- and Inter-Cluster Physicochemical Property Similarities")
plt.savefig('cluster_similarity_heatmap.png')
plt.show()
