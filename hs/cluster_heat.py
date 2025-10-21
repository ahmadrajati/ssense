
# Create a cross-tabulation for proportions
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("audio_features_recorder_clustered.csv")





# Create a heatmap of cluster counts by hour
plt.figure(figsize=(12, 8))
cluster_counts = pd.crosstab(df['filename'], df['cluster'])

sns.heatmap(cluster_counts, annot=True, fmt='d', cmap='YlOrRd', 
            cbar_kws={'label': 'Count'})
plt.title('Heatmap: Cluster Counts by Hour')
plt.xlabel('Cluster')
plt.ylabel('Hour')
plt.tight_layout()
plt.show()