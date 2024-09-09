#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:23:24 2024

@author: geet
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load the data
input_file = 'input.csv'  # Replace with your actual file path
data = pd.read_csv(input_file)

# Extract the feature columns (assuming ECFP4 columns start with 'ECFP4_')
ecfp_prefix = 'ECFP4_'
ecfp_cols = [col for col in data.columns if col.startswith(ecfp_prefix)]
if not ecfp_cols:
    raise ValueError("No ECFP4 columns found with the specified prefix.")

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(data[ecfp_cols])

# Determine the optimal number of clusters using silhouette scores
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the optimal number of clusters
optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Save the result to a new CSV file
output_file = 'output.csv'
data.to_csv(output_file, index=False)
print(f"Clustered data saved to {output_file}")
