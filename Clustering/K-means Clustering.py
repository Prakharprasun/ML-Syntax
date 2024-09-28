import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data (blobs)
X, y = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1],