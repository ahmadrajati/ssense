import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd



data = pd.read_csv("audio_features_recorder1.csv")
data2 = pd.read_csv("audio_features_recorder2.csv")

data = pd.concat([data, data2])
data.reset_index(drop=True, inplace=True)
# Generate sample data (you can replace this with your own dataset)
X = data[["rms", 	"zcr",	
    "centroid","bandwidth","rolloff","mfcc_1","mfcc_2","mfcc_3","mfcc_4","mfcc_5",
    "mfcc_6","mfcc_7","mfcc_8","mfcc_9","mfcc_10","mfcc_11","mfcc_12","mfcc_13",
    "chroma_mean","contrast_mean","tonnetz_mean"]]

# Standardize the features (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using elbow method
def find_optimal_clusters(X, max_k=15):
    wcss = []  # Within-cluster sum of squares
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal K')
    plt.show()
    
    return wcss

# Find optimal clusters
wcss = find_optimal_clusters(X_scaled)

# Perform K-means clustering with optimal k (let's choose 4 based on elbow method)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
y_pred = kmeans.fit_predict(X_scaled)

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, y_pred)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Plot the results
plt.figure(figsize=(12, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Clustered data
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

# Print cluster information
print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Labels: {y_pred[:15]}...")  # Show first 10 labels


