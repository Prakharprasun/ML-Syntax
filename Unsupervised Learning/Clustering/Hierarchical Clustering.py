import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs

# Generate sample data for clustering
X, y = make_blobs(n_samples=10, centers=3, random_state=42)

# Step 1: Compute the linkage matrix
# linkage method: 'ward', 'single', 'complete', 'average', etc.
Z = linkage(X, method='ward')

# Step 2: Plot the dendrogram
plt.figure(figsize=(8, 5))
dendrogram(Z, labels=np.arange(1, 11))  # labels for each sample
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()

# Step 3: Optional: Cut the dendrogram to form flat clusters
# Choose a threshold distance to cut the dendrogram into clusters
clusters = fcluster(Z, t=5, criterion='distance')

# Output the cluster labels
print("Cluster assignments:", clusters)
