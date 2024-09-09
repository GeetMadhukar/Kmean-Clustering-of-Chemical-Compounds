#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:44:57 2024

@author: geet
"""

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# Load data
data = pd.read_csv('input.csv') #Enter path to csv file of compounds with cluster labels

# Extract ECFP4 fingerprints and cluster labels
ecfp4_fps = data.filter(regex='ECFP4_bit_').values
cluster_labels = data['Cluster'].values
smiles = data['Smiles']  # Ensure your CSV has a 'Smiles' column

# Function to calculate MACCS fingerprints from SMILES
def calculate_maccs_fingerprints(smiles_list):
    maccs_fps = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_fps.append(list(maccs_fp))
        else:
            # Handle invalid SMILES
            maccs_fps.append([0] * 166)  # MACCS fingerprints are 166 bits long
    return np.array(maccs_fps)

# Calculate MACCS fingerprints
maccs_fps = calculate_maccs_fingerprints(smiles)

# Check the shape of maccs_fps
print("Shape of MACCS fingerprints:", maccs_fps.shape)

# Function to compute similarity matrix
def compute_similarity_matrix(fingerprints):
    if fingerprints.size == 0:
        raise ValueError("Input fingerprints array is empty.")
    return 1 - pairwise_distances(fingerprints, metric='jaccard')

# Compute similarity matrices
similarity_matrix_ecfp4 = compute_similarity_matrix(ecfp4_fps)
similarity_matrix_maccs = compute_similarity_matrix(maccs_fps)

# Function to compute intra- and inter-cluster similarities
def compute_intra_inter_cluster_similarity(matrix, labels):
    unique_clusters = np.unique(labels)
    num_clusters = len(unique_clusters)
    
    intra_cluster_similarities = np.zeros(num_clusters)
    inter_cluster_similarities = np.zeros((num_clusters, num_clusters))
    
    for i, cluster_i in enumerate(unique_clusters):
        mask_i = (labels == cluster_i)
        intra_cluster_similarities[i] = np.mean(matrix[np.ix_(mask_i, mask_i)])
        
        for j, cluster_j in enumerate(unique_clusters):
            if cluster_i != cluster_j:
                mask_j = (labels == cluster_j)
                inter_cluster_similarities[i, j] = np.mean(matrix[np.ix_(mask_i, mask_j)])
    
    return intra_cluster_similarities, inter_cluster_similarities

# Compute intra- and inter-cluster similarities
intra_sim_ecfp4, inter_sim_ecfp4 = compute_intra_inter_cluster_similarity(similarity_matrix_ecfp4, cluster_labels)
intra_sim_maccs, inter_sim_maccs = compute_intra_inter_cluster_similarity(similarity_matrix_maccs, cluster_labels)

# Function to generate and save heatmap with diagonal intra-cluster and lower triangle inter-cluster similarities
def generate_combined_heatmap(intra_sim, inter_sim, title, filename):
    num_clusters = len(intra_sim)
    
    # Create combined matrix
    combined_matrix = np.zeros((num_clusters, num_clusters))
    
    # Fill in diagonal with intra-cluster similarities
    np.fill_diagonal(combined_matrix, intra_sim)
    
    # Fill in lower triangle with inter-cluster similarities
    for i in range(num_clusters):
        for j in range(i):
            combined_matrix[i, j] = inter_sim[i, j]
    
    # Mask to keep only the lower triangle
    mask = np.triu(np.ones_like(combined_matrix, dtype=bool), k=1)
    combined_matrix = np.ma.masked_where(mask, combined_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                xticklabels=[f'Cluster {i}' for i in range(num_clusters)],  # Labels starting from 0
                yticklabels=[f'Cluster {i}' for i in range(num_clusters)],  # Labels starting from 0
                cbar_kws={'label': 'Similarity'},
                square=True,  # Ensure square aspect ratio
                linewidths=0.5,
                mask=mask)  # Apply mask directly to the heatmap
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Generate and save heatmaps for ECFP4 and MACCS
generate_combined_heatmap(intra_sim_ecfp4, inter_sim_ecfp4, 'ECFP4 Intra- and Inter-Cluster Similarities', 'ecfp4_combined_heatmap.png')
generate_combined_heatmap(intra_sim_maccs, inter_sim_maccs, 'MACCS Intra- and Inter-Cluster Similarities', 'maccs_combined_heatmap.png')

# Save correlation data
ecfp4_correlation = np.corrcoef(similarity_matrix_ecfp4)
maccs_correlation = np.corrcoef(similarity_matrix_maccs)

pd.DataFrame(ecfp4_correlation).to_csv('ecfp4_correlation.csv', index=False)
pd.DataFrame(maccs_correlation).to_csv('maccs_correlation.csv', index=False)

# Save MACCS fingerprints to CSV
num_maccs_bits = maccs_fps.shape[1]
maccs_df = pd.DataFrame(maccs_fps, columns=[f'MACCS_key_{i}' for i in range(num_maccs_bits)])
maccs_df = pd.concat([data[['ChEMBLID', 'Smiles', 'IC50_nM', 'Cluster']], maccs_df], axis=1)
maccs_df.to_csv('compounds_with_maccs_fps.csv', index=False)
