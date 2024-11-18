import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

# Sample dataset
data = {
    'AnnualIncome': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}

df = pd.DataFrame(data)

# Perform hierarchical/divisive clustering using 'ward' method
linked = linkage(df, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,orientation='top',distance_sort='descending',show_leaf_counts=True)
plt.title('Dendrogram for Divisive Clustering')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Determine clusters by cutting the dendrogram at a certain height
num_clusters = 3
clusters = cut_tree(linked, n_clusters=num_clusters).flatten()

# Add the cluster labels to the dataframe
df['Cluster'] = clusters

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis', s=100)
plt.title('Divisive Clustering of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
