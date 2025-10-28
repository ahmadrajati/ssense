import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesClassifier:
    def __init__(self, model_path, scaler_path):
        """Initialize the classifier with saved model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = ['mean', 'max']
        
    def extract_features(self, time_series_data):
        """
        Extract features from time series data
        Args:
            time_series_data: list or numpy array of time series values
        Returns:
            features: extracted features array
        """
        series = np.array(time_series_data)
        # Apply maximum constraint (values <= 0) as in training
        series = np.minimum(series, 0)
        
        # Extract features (same as training)
        features = [
            np.mean(series),    # Mean of the values
            np.max(series)      # Maximum value
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, time_series_data):
        """
        Predict cluster for time series data
        Args:
            time_series_data: list or numpy array of time series values
        Returns:
            cluster: predicted cluster (0, 1, or 2)
            probabilities: probability for each cluster
        """
        features = self.extract_features(time_series_data)
        
        # Scale features if needed (check model type)
        model_type = type(self.model).__name__
        if model_type in ['SVC', 'LogisticRegression']:
            features = self.scaler.transform(features)
        
        cluster = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return cluster, probabilities

class InfluxDBHandler:
    def __init__(self, url, token, org, bucket):
        """Initialize InfluxDB client"""
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.org = org
        
    def query_time_series_data(self, measurement, field, start_time="-1h", end_time="now()", window="1m"):
        """
        Query time series data from InfluxDB
        Args:
            measurement: measurement name to query
            field: field name to analyze
            start_time: start time for query (relative or absolute)
            end_time: end time for query
            window: aggregation window
        Returns:
            DataFrame with time series data
        """
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_time}, stop: {end_time})
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> filter(fn: (r) => r._field == "{field}")
          |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
          |> yield(name: "mean_values")
        '''
        
        try:
            result = self.client.query_api().query_data_frame(org=self.org, query=query)
            
            if result.empty:
                print(f"No data found for measurement '{measurement}', field '{field}'")
                return pd.DataFrame()
            
            # Clean and format the data
            if isinstance(result, list):
                result = pd.concat(result)
            
            # Select relevant columns and sort by time
            result = result[['_time', '_value']].sort_values('_time')
            result.columns = ['time', 'value']
            
            print(f"Retrieved {len(result)} data points from {measurement}.{field}")
            return result
            
        except Exception as e:
            print(f"Error querying InfluxDB: {e}")
            return pd.DataFrame()
    
    def write_classification_results(self, measurement, results):
        """
        Write classification results to InfluxDB
        Args:
            measurement: target measurement name
            results: list of dictionaries with classification results
        """
        from influxdb_client import Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        
        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        points = []
        for result in results:
            point = Point(measurement) \
                .tag("source_measurement", result.get("source_measurement", "")) \
                .tag("source_field", result.get("source_field", "")) \
                .field("predicted_cluster", result["cluster"]) \
                .field("probability_cluster_0", float(result["probabilities"][0])) \
                .field("probability_cluster_1", float(result["probabilities"][1])) \
                .field("probability_cluster_2", float(result["probabilities"][2])) \
                .field("confidence", float(result["confidence"])) \
                .field("mean_value", float(result["mean_value"])) \
                .field("max_value", float(result["max_value"])) \
                .time(result["timestamp"])
            
            points.append(point)
        
        try:
            write_api.write(bucket=self.bucket, record=points)
            print(f"Successfully wrote {len(points)} classification results to '{measurement}'")
        except Exception as e:
            print(f"Error writing to InfluxDB: {e}")
        finally:
            write_api.close()

def process_time_series_windows(df, classifier, window_size=10, step_size=5):
    """
    Process time series data using sliding windows
    Args:
        df: DataFrame with time series data
        classifier: TimeSeriesClassifier instance
        window_size: number of points in each window
        step_size: step between windows
    Returns:
        List of classification results
    """
    results = []
    values = df['value'].values
    times = df['time'].values
    
    if len(values) < window_size:
        print(f"Not enough data points. Need at least {window_size}, got {len(values)}")
        return results
    
    for i in range(0, len(values) - window_size + 1, step_size):
        window_data = values[i:i + window_size]
        window_time = times[i + window_size - 1]  # Use the last timestamp of the window
        
        try:
            # Predict cluster
            cluster, probabilities = classifier.predict(window_data)
            confidence = np.max(probabilities)
            
            result = {
                "timestamp": window_time,
                "cluster": int(cluster),
                "probabilities": probabilities.tolist(),
                "confidence": float(confidence),
                "mean_value": float(np.mean(window_data)),
                "max_value": float(np.max(window_data)),
                "window_size": window_size,
                "data_points": window_data.tolist()
            }
            
            results.append(result)
            print(f"Window {i//step_size + 1}: Cluster {cluster}, Confidence: {confidence:.3f}, "
                  f"Time: {window_time}")
                  
        except Exception as e:
            print(f"Error processing window starting at index {i}: {e}")
    
    return results

def main():
    # Configuration - Update these with your InfluxDB details
    INFLUX_CONFIG = {
        "url": "http://localhost:8086",  # Your InfluxDB URL
        "token": "your-token-here",      # Your InfluxDB token
        "org": "your-org",               # Your organization
        "bucket": "your-bucket"          # Your bucket name
    }
    
    # Model paths
    MODEL_PATH = "best_classification_model_random_forest.pkl"  # Update with your actual model name
    SCALER_PATH = "scaler.pkl"
    
    # Source data configuration
    SOURCE_MEASUREMENT = "sensor_data"  # Your source measurement
    SOURCE_FIELD = "value"              # Your field to analyze
    TARGET_MEASUREMENT = "time_series_classification"  # Target for results
    
    # Time window configuration
    QUERY_START_TIME = "-1h"  # Query last 1 hour of data
    WINDOW_SIZE = 10          # Same as training (10 time points)
    STEP_SIZE = 5             # Step between windows (50% overlap)
    
    try:
        # Initialize classifier
        print("Loading trained model and scaler...")
        classifier = TimeSeriesClassifier(MODEL_PATH, SCALER_PATH)
        print("Model loaded successfully!")
        
        # Initialize InfluxDB handler
        print("Connecting to InfluxDB...")
        influx_handler = InfluxDBHandler(
            url=INFLUX_CONFIG["url"],
            token=INFLUX_CONFIG["token"],
            org=INFLUX_CONFIG["org"],
            bucket=INFLUX_CONFIG["bucket"]
        )
        print("Connected to InfluxDB!")
        
        # Query time series data
        print(f"Querying data from {SOURCE_MEASUREMENT}.{SOURCE_FIELD}...")
        time_series_data = influx_handler.query_time_series_data(
            measurement=SOURCE_MEASUREMENT,
            field=SOURCE_FIELD,
            start_time=QUERY_START_TIME
        )
        
        if time_series_data.empty:
            print("No data retrieved. Exiting.")
            return
        
        # Process data with sliding windows
        print(f"Processing data with window size {WINDOW_SIZE}, step size {STEP_SIZE}...")
        classification_results = process_time_series_windows(
            time_series_data, 
            classifier, 
            window_size=WINDOW_SIZE, 
            step_size=STEP_SIZE
        )
        
        if not classification_results:
            print("No classification results generated. Exiting.")
            return
        
        # Add source information to results
        for result in classification_results:
            result["source_measurement"] = SOURCE_MEASUREMENT
            result["source_field"] = SOURCE_FIELD
        
        # Write results to InfluxDB
        print(f"Writing {len(classification_results)} results to {TARGET_MEASUREMENT}...")
        influx_handler.write_classification_results(TARGET_MEASUREMENT, classification_results)
        
        # Print summary
        clusters = [r["cluster"] for r in classification_results]
        confidences = [r["confidence"] for r in classification_results]
        
        print("\n" + "="*50)
        print("CLASSIFICATION SUMMARY")
        print("="*50)
        print(f"Total windows processed: {len(classification_results)}")
        print(f"Cluster distribution:")
        for cluster in range(3):
            count = clusters.count(cluster)
            percentage = (count / len(clusters)) * 100
            print(f"  Cluster {cluster}: {count} windows ({percentage:.1f}%)")
        print(f"Average confidence: {np.mean(confidences):.3f}")
        print(f"Results written to: {TARGET_MEASUREMENT}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

def single_prediction_example():
    """Example of making a single prediction with new data"""
    MODEL_PATH = "best_classification_model_random_forest.pkl"
    SCALER_PATH = "scaler.pkl"
    
    classifier = TimeSeriesClassifier(MODEL_PATH, SCALER_PATH)
    
    # Example time series data (similar to your training data)
    example_data = np.random.normal(loc=-40, scale=15, size=10)
    example_data = np.minimum(example_data, 0)  # Apply constraint
    
    cluster, probabilities = classifier.predict(example_data)
    
    print("\n" + "="*50)
    print("SINGLE PREDICTION EXAMPLE")
    print("="*50)
    print(f"Input data: {example_data.round(2)}")
    print(f"Features - Mean: {np.mean(example_data):.2f}, Max: {np.max(example_data):.2f}")
    print(f"Predicted cluster: {cluster}")
    print(f"Probabilities: Cluster 0: {probabilities[0]:.3f}, "
          f"Cluster 1: {probabilities[1]:.3f}, Cluster 2: {probabilities[2]:.3f}")
    print(f"Confidence: {np.max(probabilities):.3f}")

if __name__ == "__main__":
    # Run the main processing pipeline
    main()
    
    # Uncomment to run single prediction example
    # single_prediction_example()