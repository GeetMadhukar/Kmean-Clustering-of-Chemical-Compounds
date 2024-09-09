**K-Means Clustering of Chemical Compounds using ECFP4 Fingerprints**


**Overview**
This project implements K-means clustering for chemical compounds based on their ECFP4 (Extended-Connectivity Fingerprints) fingerprints. The goal is to group compounds with similar molecular structures and properties into clusters. K-means is a popular unsupervised learning algorithm used in cheminformatics to explore compound similarity, diversity, and structure-activity relationships.

**Key Features:**
ECFP4 Fingerprint Extraction: The code automatically identifies features (columns) in the input data corresponding to ECFP4 fingerprints.
Data Standardization: Before clustering, the input data is standardized using StandardScaler from scikit-learn to ensure that all features contribute equally.
Silhouette Analysis: The optimal number of clusters is determined using silhouette scores, which quantify how well the compounds fit within their assigned clusters and how distinct the clusters are.
K-means Clustering: The algorithm clusters compounds into an optimal number of groups based on the structural similarities captured in the ECFP4 fingerprints.
Result Export: The output is saved as a CSV file, with an additional column indicating the cluster label for each compound.

**How It Works**
1. Input Data
The script reads compound data from an input CSV file. The input should contain ECFP4 fingerprints as separate columns, prefixed with "ECFP4_". These fingerprints are used as features for clustering. Ensure that your input data is correctly formatted.

2. Data Preprocessing
The ECFP4 fingerprint data is standardized using z-score normalization, ensuring that each feature has a mean of 0 and a standard deviation of 1. This step is crucial for improving the performance of the K-means algorithm.

3. Determining Optimal Clusters
The script evaluates clustering performance for different numbers of clusters (from 2 to 10) using silhouette scores. The silhouette score is a measure of how similar a compound is to its own cluster compared to other clusters. The number of clusters that yields the highest silhouette score is selected as the optimal number.

4. K-Means Clustering
Once the optimal number of clusters is determined, the K-means algorithm groups the compounds into clusters. Each compound is assigned a cluster label, indicating which cluster it belongs to based on the similarity of its ECFP4 fingerprints.

5. Output
The resulting clustered dataset is saved as a new CSV file with an additional Cluster column containing the cluster labels. This file can be used for further analysis, such as identifying structure-activity relationships or exploring compound diversity.

**How to Run**
1. Install the required Python packages:

pip install pandas scikit-learn

2. Prepare the input CSV file with ECFP4 fingerprint columns, prefixed by ECFP4_.

3. Run the script:
   
python kmeans_clustering.py

The output CSV file will be saved as output.csv, containing the original data along with an additional Cluster column.
