import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def generate_clustered_time_series_data(n_samples=300, series_length=10):
    """
    Generate balanced data with three clusters where each sample has 10 time points
    Extract multiple features from each time series
    """
    # Calculate samples per cluster (balanced)
    samples_per_cluster = n_samples // 3
    
    # Define cluster parameters for generating the 10-point series
    cluster_params = {
        0: {'mean': -65, 'std': 25},   # Cluster 0: Low values, decreasing
        1: {'mean': -40, 'std': 30},    # Cluster 1: Medium values, stable
        2: {'mean': -15, 'std': 20}     # Cluster 2: High values, increasing
    }
    
    all_features = []
    all_labels = []
    all_raw_series = []
    
    for cluster_id in range(3):
        params = cluster_params[cluster_id]
        
        for i in range(samples_per_cluster):
            # Generate 10 random values for this sample
            base_series = np.random.normal(
                loc=params['mean'], 
                scale=params['std'], 
                size=series_length
            )
            
            # Add a trend to make patterns more distinctive
            series = base_series 
            
            # Apply maximum constraint (values <= 0)
            series = np.minimum(series, 0)
            
            # Extract multiple features from the series
            features = [
                np.mean(series),           # Mean of the 10 values
                np.max(series)          # Maximum value
            ]
            
            all_features.append(features)
            all_labels.append(cluster_id)
            all_raw_series.append(series)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    raw_series = np.array(all_raw_series)
    
    return X, y, raw_series

# Generate the data
print("Generating time series data with multiple features...")
X, y, raw_series = generate_clustered_time_series_data(n_samples=300, series_length=10)

# Create feature names
feature_names = [
    'mean', 'max'
]

# Create DataFrame for analysis
df = pd.DataFrame(X, columns=feature_names)
df['true_cluster'] = y

print(f"Generated {len(df)} samples")
print(f"Cluster distribution:\n{df['true_cluster'].value_counts().sort_index()}")

# Display cluster statistics for key features
print("\nCluster Statistics for Key Features:")
key_features = ['mean', 'max']
cluster_stats = df.groupby('true_cluster')[key_features].agg(['mean', 'max']).round(2)
print(cluster_stats)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

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

# Create and save individual charts

# 1. Distribution of mean feature by cluster
plt.figure(figsize=(8, 6))
for cluster in range(3):
    cluster_data = df[df['true_cluster'] == cluster]['mean']
    plt.hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=20, density=True)
plt.title('Mean Feature Distribution by Cluster')
plt.xlabel('Mean Value')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('mean_feature_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Distribution of max feature by cluster
plt.figure(figsize=(8, 6))
for cluster in range(3):
    cluster_data = df[df['true_cluster'] == cluster]['max']
    plt.hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=20, density=True)
plt.title('Max Feature Distribution by Cluster')
plt.xlabel('Max Value')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('max_feature_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature importance (Random Forest)
plt.figure(figsize=(8, 6))
if hasattr(trained_models['Random Forest'], 'feature_importances_'):
    feature_importance = trained_models['Random Forest'].feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], color='lightseagreen')
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx], rotation=45)
    plt.title('Random Forest Feature Importance')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Model performance comparison
plt.figure(figsize=(8, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 5-7. Confusion matrices for each model
for name, result in results.items():
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2'], 
                yticklabels=['0', '1', '2'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. PCA visualization (2D projection)
plt.figure(figsize=(8, 6))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)
explained_variance = pca.explained_variance_ratio_

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='viridis', alpha=0.7, s=50)
plt.colorbar(label='True Cluster')
plt.title(f'PCA Visualization\n(Explained Variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f})')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Correlation heatmap of features
plt.figure(figsize=(6, 5))
correlation_matrix = df[feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Example time series from each cluster
plt.figure(figsize=(8, 6))
for cluster in range(3):
    cluster_series = raw_series[y == cluster]
    # Plot a few examples
    for i in range(min(2, len(cluster_series))):
        plt.plot(cluster_series[i], alpha=0.7, label=f'Cluster {cluster}' if i == 0 else "")
plt.title('Example Time Series')
plt.xlabel('Time Point')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.savefig('example_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# 13. Prediction probabilities distribution
plt.figure(figsize=(8, 6))
if 'probabilities' in results[best_model_name]:
    proba = results[best_model_name]['probabilities']
    for cluster in range(3):
        plt.hist(proba[:, cluster], alpha=0.6, label=f'Cluster {cluster}', bins=20)
    plt.title(f'{best_model_name} Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_probabilities.png', dpi=300, bbox_inches='tight')
    plt.close()

# 14. Misclassification analysis in PCA space
plt.figure(figsize=(8, 6))
best_predictions = results[best_model_name]['predictions']
misclassified = (y_test != best_predictions)

plt.scatter(X_pca[~misclassified, 0], X_pca[~misclassified, 1], 
           c='green', label='Correct', s=30, alpha=0.6)
plt.scatter(X_pca[misclassified, 0], X_pca[misclassified, 1], 
           c='red', label='Misclassified', s=60, alpha=0.8)
plt.title('Misclassification Analysis (PCA)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.tight_layout()
plt.savefig('misclassification_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save detailed results to Excel
print("\nSaving detailed results to Excel...")

# Create results DataFrame
results_df = pd.DataFrame(X_test, columns=feature_names)
results_df['true_cluster'] = y_test
results_df['best_model_prediction'] = results[best_model_name]['predictions']
results_df['correct'] = (y_test == results[best_model_name]['predictions'])

# Add predictions from all models
for name in classifiers.keys():
    results_df[f'{name}_prediction'] = results[name]['predictions']

# Save raw time series data too
raw_series_df = pd.DataFrame(raw_series, columns=[f'time_point_{i+1}' for i in range(10)])
raw_series_df['true_cluster'] = y

# Save to Excel with multiple sheets
with pd.ExcelWriter('classification_results_time_series.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Test_Results', index=False)
    raw_series_df.to_excel(writer, sheet_name='Raw_Time_Series', index=False)
    df.describe().to_excel(writer, sheet_name='Feature_Statistics')

# Print final summary
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"Best model: {best_model_name}")
print(f"Best model accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"\nGenerated features from 10-point time series:")
for i, feature in enumerate(feature_names):
    print(f"  {i+1:2d}. {feature}")
print(f"\nFiles saved:")
print("- Individual chart files (PNG format)")
print("- classification_results_time_series.xlsx (Detailed results with multiple sheets)")
print(f"- best_classification_model_{best_model_name.replace(' ', '_').lower()}.pkl (Best model)")
print("- scaler.pkl (Feature scaler)")

# Example of how to use the saved model for new predictions
print("\n" + "="*50)
print("EXAMPLE: PREDICTING NEW SAMPLES")
print("="*50)

# Load the saved model and scaler
loaded_model = joblib.load(f'best_classification_model_{best_model_name.replace(" ", "_").lower()}.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Create new sample time series and extract features
new_time_series = [
    np.random.normal(loc=-60, scale=12, size=10),  # Should be cluster 0
    np.random.normal(loc=-35, scale=18, size=10),  # Should be cluster 1
    np.random.normal(loc=-20, scale=8, size=10),   # Should be cluster 2
]

new_features = []
for series in new_time_series:
    series = np.minimum(series, 0)  # Apply max constraint
    features = [
        np.mean(series), np.max(series)
    ]
    new_features.append(features)

new_features = np.array(new_features)

# Preprocess and predict
if best_model_name in ['SVM', 'Logistic Regression']:
    new_features_scaled = loaded_scaler.transform(new_features)
    new_predictions = loaded_model.predict(new_features_scaled)
    new_probabilities = loaded_model.predict_proba(new_features_scaled)
else:
    new_predictions = loaded_model.predict(new_features)
    new_probabilities = loaded_model.predict_proba(new_features)

print("New sample predictions:")
for i, (series, pred, proba) in enumerate(zip(new_time_series, new_predictions, new_probabilities)):
    print(f"Sample {i+1}: Mean={np.mean(series):.1f}, Max={np.max(series):.1f} -> Cluster {pred}")
    print(f"  Probabilities: Cluster 0: {proba[0]:.3f}, Cluster 1: {proba[1]:.3f}, Cluster 2: {proba[2]:.3f}")
    print(f"  Time series: {series.round(1)}")