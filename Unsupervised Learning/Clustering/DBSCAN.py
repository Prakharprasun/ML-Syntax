import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate synthetic dataset (moons dataset)
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=50, color='gray')
plt.title('Data Points')
plt.show()

# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)  # eps is the ε parameter, min_samples is MinPts
y_dbscan = dbscan.fit_predict(X)

# Plot DBSCAN clusters
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='plasma', s=50)
plt.title('DBSCAN Clustering')
plt.show()

# Identify core, border, and noise points
core_samples_mask = np.zeros_like(y_dbscan, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
outliers = (y_dbscan == -1)

# Show the cluster and outlier count
print(f"Number of clusters: {len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)}")
print(f"Number of outliers: {np.sum(outliers)}")


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns

# Load Iris dataset
iris = load_iris()
X = iris.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

# Convert to DataFrame for better visualization
iris_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
iris_df['cluster'] = y_dbscan

# Visualize clusters using a pairplot
sns.pairplot(iris_df, hue='cluster', palette='Set1')
plt.show()

# Show number of clusters and outliers
n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_outliers = np.sum(y_dbscan == -1)
print(f'Estimated number of clusters: {n_clusters}')
print(f'Number of noise points: {n_outliers}')
