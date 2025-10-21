# Create a cross-tabulation for proportions
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("audio_features_recorder_clustered.csv")

cluster_by_hour = pd.crosstab(df['filename'], df['cluster'], normalize='index')

# Stacked bar chart
plt.figure(figsize=(14, 8))
cluster_by_hour.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Proportion of Clusters in Each Hour')
plt.xlabel('Hour')
plt.ylabel('Proportion')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()