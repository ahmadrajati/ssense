import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def generate_clustered_data(n_samples=300):
    """
    Generate balanced data with three clusters for volume and peak features
    """
    # Calculate samples per cluster (balanced)
    samples_per_cluster = n_samples // 3
    
    # Generate VOLUME data for each cluster (main feature)
    volume_cluster1 = np.random.normal(loc=-65, scale=15, size=samples_per_cluster)
    volume_cluster2 = np.random.normal(loc=-40, scale=20, size=samples_per_cluster)
    volume_cluster3 = np.random.normal(loc=-15, scale=10, size=samples_per_cluster)
    
    # Generate PEAK data that is correlated with volume
    # Generally, higher volume (less negative) corresponds to higher peak values
    peak_cluster1 = np.random.normal(loc=30, scale=8, size=samples_per_cluster)   # Lower volume = lower peaks
    peak_cluster2 = np.random.normal(loc=50, scale=12, size=samples_per_cluster)  # Medium volume = medium peaks
    peak_cluster3 = np.random.normal(loc=70, scale=6, size=samples_per_cluster)   # Higher volume = higher peaks
    
    # Apply maximum constraint to volume (values <= 0)
    volume_cluster1 = np.minimum(volume_cluster1, 0)
    volume_cluster2 = np.minimum(volume_cluster2, 0)
    volume_cluster3 = np.minimum(volume_cluster3, 0)
    
    # Ensure peaks are positive and have reasonable bounds
    peak_cluster1 = np.maximum(peak_cluster1, 10)
    peak_cluster2 = np.maximum(peak_cluster2, 20)
    peak_cluster3 = np.maximum(peak_cluster3, 50)
    
    # Combine all data
    all_volumes = np.concatenate([volume_cluster1, volume_cluster2, volume_cluster3])
    all_peaks = np.concatenate([peak_cluster1, peak_cluster2, peak_cluster3])
    
    # Create labels (0, 1, 2 for the three clusters)
    true_labels = np.concatenate([
        np.zeros(samples_per_cluster),  # Cluster 1
        np.ones(samples_per_cluster),   # Cluster 2
        np.full(samples_per_cluster, 2) # Cluster 3
    ])
    
    # Create feature matrix with both volume and peak features
    X = np.column_stack([all_volumes, all_peaks])
    
    return X, true_labels, all_volumes, all_peaks

# Generate the data
print("Generating clustered data with volume and peak features...")
X, y, volumes, peaks = generate_clustered_data(n_samples=300)

# Create DataFrame for analysis
df = pd.DataFrame({
    'volume': volumes,
    'peak': peaks,
    'true_cluster': y.astype(int)
})

print(f"Generated {len(df)} samples")
print(f"Cluster distribution:\n{df['true_cluster'].value_counts().sort_index()}")

# Display cluster statistics
print("\nCluster Statistics:")
cluster_stats = df.groupby('true_cluster').agg({
    'volume': ['mean', 'std', 'min', 'max'],
    'peak': ['mean', 'std', 'min', 'max']
}).round(2)
print(cluster_stats)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Train and evaluate classifiers
results = {}
trained_models = {}

print("\n" + "="*50)
print("TRAINING CLASSIFICATION MODELS")
print("="*50)

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    if name == 'SVM' or name == 'Logistic Regression':
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': clf,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    trained_models[name] = clf
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

# Save the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
joblib.dump(best_model, f'best_classification_model_{best_model_name.replace(" ", "_").lower()}.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
print("Best model saved!")

# Create comprehensive visualizations
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))

# 1. Volume distribution by cluster
plt.subplot(3, 4, 1)
for cluster in range(3):
    cluster_data = df[df['true_cluster'] == cluster]['volume']
    plt.hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=20, density=True)
plt.title('Volume Distribution by Cluster')
plt.xlabel('Volume')
plt.ylabel('Density')
plt.legend()
plt.axvline(x=-65, color='blue', linestyle='--', alpha=0.3)
plt.axvline(x=-40, color='orange', linestyle='--', alpha=0.3)
plt.axvline(x=-15, color='green', linestyle='--', alpha=0.3)

# 2. Peak distribution by cluster
plt.subplot(3, 4, 2)
for cluster in range(3):
    cluster_data = df[df['true_cluster'] == cluster]['peak']
    plt.hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=20, density=True)
plt.title('Peak Distribution by Cluster')
plt.xlabel('Peak')
plt.ylabel('Density')
plt.legend()

# 3. 2D Feature space visualization (Volume vs Peak)
plt.subplot(3, 4, 3)
scatter = plt.scatter(df['volume'], df['peak'], c=df['true_cluster'], 
                     cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='True Cluster')
plt.title('2D Feature Space (Volume vs Peak)')
plt.xlabel('Volume')
plt.ylabel('Peak')

# 4. Model performance comparison
plt.subplot(3, 4, 4)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')

# 5-7. Confusion matrices for each model
for i, (name, result) in enumerate(results.items()):
    plt.subplot(3, 4, 5 + i)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2'], 
                yticklabels=['0', '1', '2'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

# 8. Feature importance (for Random Forest)
plt.subplot(3, 4, 8)
if hasattr(trained_models['Random Forest'], 'feature_importances_'):
    feature_importance = trained_models['Random Forest'].feature_importances_
    features = ['Volume', 'Peak']
    plt.bar(features, feature_importance, color='lightseagreen')
    plt.title('Random Forest Feature Importance')
    plt.ylabel('Importance')
    for i, v in enumerate(feature_importance):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 9. Decision boundaries for best model
plt.subplot(3, 4, 9)
# Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))

if best_model_name in ['SVM', 'Logistic Regression']:
    # Scale the mesh grid for models that require scaling
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = best_model.predict(mesh_points_scaled)
else:
    Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
    
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], 
                     c=y_test, cmap='viridis', edgecolors='black', s=50)
plt.colorbar(scatter, label='True Cluster')
plt.title(f'Decision Boundaries - {best_model_name}')
plt.xlabel('Volume')
plt.ylabel('Peak')

# 10. Prediction probabilities distribution
plt.subplot(3, 4, 10)
if 'probabilities' in results[best_model_name]:
    proba = results[best_model_name]['probabilities']
    for cluster in range(3):
        plt.hist(proba[:, cluster], alpha=0.6, label=f'Cluster {cluster}', bins=20)
    plt.title(f'{best_model_name} Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()

# 11. Correlation heatmap
plt.subplot(3, 4, 11)
correlation_matrix = df[['volume', 'peak', 'true_cluster']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Feature Correlation Heatmap')

# 12. Misclassification analysis
plt.subplot(3, 4, 12)
best_predictions = results[best_model_name]['predictions']
misclassified = (y_test != best_predictions)
plt.scatter(X_test[misclassified, 0], X_test[misclassified, 1], 
           c='red', label='Misclassified', s=60, alpha=0.7)
plt.scatter(X_test[~misclassified, 0], X_test[~misclassified, 1], 
           c='green', label='Correct', s=30, alpha=0.6)
plt.title('Misclassification Analysis')
plt.xlabel('Volume')
plt.ylabel('Peak')
plt.legend()

plt.tight_layout()
plt.savefig('classification_analysis_volume_peak.png', dpi=300, bbox_inches='tight')
plt.show()

# Save detailed results to Excel
print("\nSaving detailed results to Excel...")
results_df = pd.DataFrame({
    'volume': X_test[:, 0],
    'peak': X_test[:, 1],
    'true_cluster': y_test,
    'best_model_prediction': best_predictions,
    'correct': (y_test == best_predictions)
})

# Add predictions from all models
for name in classifiers.keys():
    results_df[f'{name}_prediction'] = results[name]['predictions']

results_df.to_excel('classification_results_volume_peak.xlsx', index=False)

# Print final summary
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"Best model: {best_model_name}")
print(f"Best model accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"\nFeature relationships:")
print(f"- Cluster 0: Low volume (~-65), Low peaks (~30)")
print(f"- Cluster 1: Medium volume (~-40), Medium peaks (~50)") 
print(f"- Cluster 2: High volume (~-15), High peaks (~70)")
print(f"\nFiles saved:")
print("- classification_analysis_volume_peak.png (Comprehensive visualizations)")
print("- classification_results_volume_peak.xlsx (Detailed predictions)")
print(f"- best_classification_model_{best_model_name.replace(' ', '_').lower()}.pkl (Best model)")
print("- scaler.pkl (Feature scaler)")

# Example of how to use the saved model for new predictions
print("\n" + "="*50)
print("EXAMPLE: PREDICTING NEW SAMPLES")
print("="*50)

# Load the saved model and scaler
loaded_model = joblib.load(f'best_classification_model_{best_model_name.replace(" ", "_").lower()}.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Create new sample data (volume, peak)
new_samples = np.array([
    [-60, 35],   # Should be close to cluster 0
    [-35, 45],   # Should be close to cluster 1  
    [-20, 65],   # Should be close to cluster 2
    [-50, 40]    # In between cluster 0 and 1
])

# Preprocess and predict
if best_model_name in ['SVM', 'Logistic Regression']:
    new_samples_scaled = loaded_scaler.transform(new_samples)
    new_predictions = loaded_model.predict(new_samples_scaled)
    new_probabilities = loaded_model.predict_proba(new_samples_scaled)
else:
    new_predictions = loaded_model.predict(new_samples)
    new_probabilities = loaded_model.predict_proba(new_samples)

print("New sample predictions:")
for i, (sample, pred, proba) in enumerate(zip(new_samples, new_predictions, new_probabilities)):
    print(f"Sample {i+1}: Volume={sample[0]:.1f}, Peak={sample[1]:.1f} -> Cluster {pred}")
    print(f"  Probabilities: Cluster 0: {proba[0]:.3f}, Cluster 1: {proba[1]:.3f}, Cluster 2: {proba[2]:.3f}")