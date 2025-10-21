# Create a cross-tabulation for proportions
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("audio_features_recorder_clustered.csv")



# Create a facet grid for detailed analysis
g = sns.FacetGrid(df, col='cluster', col_wrap=3, height=4, aspect=1.5)
g.map(sns.histplot, 'filename', bins=24, kde=True)
g.set_axis_labels('Hour', 'Count')
g.set_titles('Cluster {col_name}')
plt.suptitle('Distribution of Hours for Each Cluster', y=1.02)
plt.tight_layout()
plt.show()