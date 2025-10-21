import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def kmeans_clustering(data, n_clusters=None, auto_select=True):
    """
    Perform K-means clustering on a dataset
    
    Parameters:
    data: numpy array or pandas DataFrame
    n_clusters: number of clusters (if None, will use elbow method)
    auto_select: whether to automatically select optimal k
    """
    
    # Convert to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Automatically select optimal k if requested
    if auto_select and n_clusters is None:
        n_clusters = find_optimal_k(data_scaled)
    
    # If n_clusters is still None, use a default
    if n_clusters is None:
        n_clusters = 3
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    
    # Add cluster labels to original data if it was a DataFrame
    if isinstance(data, pd.DataFrame):
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
    else:
        data_with_clusters = np.column_stack([data, labels])
    
    return labels, kmeans, data_with_clusters




data = pd.read_csv("audio_features_recorder1.csv")
data2 = pd.read_csv("audio_features_recorder2.csv")

data = pd.concat([data, data2])
data.reset_index(drop=True, inplace=True)
# Generate sample data (you can replace this with your own dataset)
X = data[["rms", 	"zcr",	
    "centroid","bandwidth","rolloff","mfcc_1","mfcc_2","mfcc_3","mfcc_4","mfcc_5",
    "mfcc_6","mfcc_7","mfcc_8","mfcc_9","mfcc_10","mfcc_11","mfcc_12","mfcc_13",
    "chroma_mean","contrast_mean","tonnetz_mean"]]


filenames = data["file"].apply(lambda x:x.split("-")[1].split("_")[0])

# Perform clustering
labels, kmeans_model, clustered_data = kmeans_clustering(data=X, n_clusters=8)
clustered_data = pd.DataFrame(clustered_data,columns=["rms", 	"zcr",	
    "centroid","bandwidth","rolloff","mfcc_1","mfcc_2","mfcc_3","mfcc_4","mfcc_5",
    "mfcc_6","mfcc_7","mfcc_8","mfcc_9","mfcc_10","mfcc_11","mfcc_12","mfcc_13",
    "chroma_mean","contrast_mean","tonnetz_mean","cluster"])

clustered_data["filename"] = filenames

clustered_data.to_csv("audio_features_recorder_clustered.csv", index=False)

print(f"Cluster labels: {labels}")
print(f"Cluster centers shape: {kmeans_model.cluster_centers_.shape}")