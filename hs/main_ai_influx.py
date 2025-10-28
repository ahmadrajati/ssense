import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
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
    def __init__(self, host='localhost', port=8086, username='', password='', database='iot_data'):
        """Initialize InfluxDB client"""
        self.client = InfluxDBClient(host=host, port=port, username=username, password=password)
        self.database = database
        
        # Switch to the specified database
        self.client.switch_database(database)
        print(f"Connected to InfluxDB at {host}:{port}, database: {database}")
    
    def query_volume_data(self, start_time="-1h", end_time="now()", limit=1000):
        """
        Query sound volume data from InfluxDB
        Args:
            start_time: start time for query (relative or absolute)
            end_time: end time for query
            limit: maximum number of points to retrieve
        Returns:
            DataFrame with time series data
        """
        query = f'''
        SELECT "volume", "peak" 
        FROM "sound_volume" 
        WHERE time >= {start_time} AND time <= {end_time}
        ORDER BY time DESC
        LIMIT {limit}
        '''
        
        try:
            print(f"Executing query: {query}")
            result = self.client.query(query)
            
            if not result:
                print("No data found in sound_volume measurement")
                return pd.DataFrame()
            
            # Convert to DataFrame
            points = list(result.get_points(measurement='sound_volume'))
            
            if not points:
                print("No data points found")
                return pd.DataFrame()
                
            df = pd.DataFrame(points)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')  # Sort by time ascending
            
            print(f"Retrieved {len(df)} data points from sound_volume")
            return df
            
        except Exception as e:
            print(f"Error querying InfluxDB: {e}")
            return pd.DataFrame()
    
    def write_classification_results(self, results):
        """
        Write classification results to InfluxDB in sound_classification measurement
        Args:
            results: list of dictionaries with classification results
        """
        json_body = []
        
        for result in results:
            point = {
                "measurement": "sound_classification",
                #"time": result["timestamp"],
                "tags": {
                    "source_measurement": "sound_volume",
                    "predicted_cluster": f"cluster_{result['cluster']}"
                },
                "fields": {
                    "cluster": int(result["cluster"]),
                    "probability_0": float(result["probabilities"][0]),
                    "probability_1": float(result["probabilities"][1]),
                    "probability_2": float(result["probabilities"][2]),
                    "confidence": float(result["confidence"]),
                    "mean_volume": float(result["mean_value"]),
                    "max_volume": float(result["max_value"]),
                    "window_size": int(result["window_size"])
                }
            }
            json_body.append(point)
        
        try:
            self.client.write_points(json_body)
            print(f"Successfully wrote {len(json_body)} classification results to 'sound_classification'")
            return True
        except Exception as e:
            print(f"Error writing to InfluxDB: {e}")
            return False

def process_volume_windows(df, classifier, window_size=10, step_size=5, use_volume=True):
    """
    Process volume data using sliding windows
    Args:
        df: DataFrame with volume data
        classifier: TimeSeriesClassifier instance
        window_size: number of points in each window
        step_size: step between windows
        use_volume: if True use 'volume' field, if False use 'peak' field
    Returns:
        List of classification results
    """
    results = []
    
    field_name = 'volume' if use_volume else 'peak'
    values = df[field_name].values
    times = df['time'].values
    
    if len(values) < window_size:
        print(f"Not enough data points. Need at least {window_size}, got {len(values)}")
        return results
    
    print(f"Processing {len(values)} data points with window size {window_size}, step size {step_size}")
    print(f"Using field: {field_name}")
    
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
                "data_points": window_data.tolist(),
                "field_used": field_name
            }
            
            results.append(result)
            print(f"Window {i//step_size + 1}: Cluster {cluster}, Confidence: {confidence:.3f}, "
                  f"Time: {window_time}, Mean: {np.mean(window_data):.2f}, Max: {np.max(window_data):.2f}")
                  
        except Exception as e:
            print(f"Error processing window starting at index {i}: {e}")
    
    return results

def main():
    # Configuration - Update these if your InfluxDB requires authentication
    INFLUX_CONFIG = {
        "host": "localhost",
        "port": 8086,
        "username": "",  # Leave empty if no authentication
        "password": "",  # Leave empty if no authentication
        "database": "iot_data"
    }
    
    # Model paths - update with your actual model filename
    MODEL_PATH = "best_classification_model_random_forest.pkl"  # Update with your actual model name
    SCALER_PATH = "scaler.pkl"
    
    # Processing configuration
    QUERY_START_TIME = "-1h"  # Query last 1 hour of data
    WINDOW_SIZE = 10          # Same as training (10 time points)
    STEP_SIZE = 5             # Step between windows (50% overlap)
    USE_VOLUME_FIELD = True   # Set to False to use 'peak' field instead of 'volume'
    
    try:
        # Initialize classifier
        print("Loading trained model and scaler...")
        classifier = TimeSeriesClassifier(MODEL_PATH, SCALER_PATH)
        print(f"Model type: {type(classifier.model).__name__}")
        print("Model loaded successfully!")
        
        # Initialize InfluxDB handler
        print("Connecting to InfluxDB...")
        influx_handler = InfluxDBHandler(
            host=INFLUX_CONFIG["host"],
            port=INFLUX_CONFIG["port"],
            username=INFLUX_CONFIG["username"],
            password=INFLUX_CONFIG["password"],
            database=INFLUX_CONFIG["database"]
        )
        
        # Query volume data
        print(f"Querying sound volume data from last hour...")
        volume_data = influx_handler.query_volume_data(
            start_time=QUERY_START_TIME,
            limit=1000
        )
        
        if volume_data.empty:
            print("No volume data retrieved. Exiting.")
            return
        
        print(f"Data columns: {volume_data.columns.tolist()}")
        print(f"Time range: {volume_data['time'].min()} to {volume_data['time'].max()}")
        print(f"Volume statistics - Mean: {volume_data['volume'].mean():.2f}, "
              f"Max: {volume_data['volume'].max():.2f}, Min: {volume_data['volume'].min():.2f}")
        
        if 'peak' in volume_data.columns:
            print(f"Peak statistics - Mean: {volume_data['peak'].mean():.2f}, "
                  f"Max: {volume_data['peak'].max():.2f}, Min: {volume_data['peak'].min():.2f}")
        
        # Process data with sliding windows
        print(f"\nProcessing data with window size {WINDOW_SIZE}, step size {STEP_SIZE}...")
        classification_results = process_volume_windows(
            volume_data, 
            classifier, 
            window_size=WINDOW_SIZE, 
            step_size=STEP_SIZE,
            use_volume=USE_VOLUME_FIELD
        )
        
        if not classification_results:
            print("No classification results generated. Exiting.")
            return
        
        # Write results to InfluxDB
        print(f"\nWriting {len(classification_results)} results to sound_classification...")
        success = influx_handler.write_classification_results(classification_results)
        
        # Print summary
        clusters = [r["cluster"] for r in classification_results]
        confidences = [r["confidence"] for r in classification_results]
        
        print("\n" + "="*50)
        print("CLASSIFICATION SUMMARY")
        print("="*50)
        print(f"Total windows processed: {len(classification_results)}")
        print(f"Field used: {classification_results[0]['field_used']}")
        print(f"Cluster distribution:")
        for cluster in range(3):
            count = clusters.count(cluster)
            percentage = (count / len(clusters)) * 100 if clusters else 0
            print(f"  Cluster {cluster}: {count} windows ({percentage:.1f}%)")
        print(f"Average confidence: {np.mean(confidences):.3f}")
        print(f"Results written to: sound_classification measurement")
        
        if success:
            print("All results successfully written to InfluxDB!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

def test_single_prediction():
    """Test making a single prediction with sample data"""
    MODEL_PATH = "best_classification_model_random_forest.pkl"
    SCALER_PATH = "scaler.pkl"
    
    try:
        classifier = TimeSeriesClassifier(MODEL_PATH, SCALER_PATH)
        
        # Create sample volume data (similar to what you'd get from InfluxDB)
        # Using negative values as per your training data constraint
        sample_volume_data = np.random.normal(loc=-40, scale=15, size=10)
        sample_volume_data = np.minimum(sample_volume_data, 0)  # Apply constraint
        
        cluster, probabilities = classifier.predict(sample_volume_data)
        
        print("\n" + "="*50)
        print("SINGLE PREDICTION TEST")
        print("="*50)
        print(f"Sample volume data: {sample_volume_data.round(2)}")
        print(f"Features - Mean: {np.mean(sample_volume_data):.2f}, Max: {np.max(sample_volume_data):.2f}")
        print(f"Predicted cluster: {cluster}")
        print(f"Probabilities: Cluster 0: {probabilities[0]:.3f}, "
              f"Cluster 1: {probabilities[1]:.3f}, Cluster 2: {probabilities[2]:.3f}")
        print(f"Confidence: {np.max(probabilities):.3f}")
        
    except Exception as e:
        print(f"Error in test prediction: {e}")

if __name__ == "__main__":
    # Run the main processing pipeline
    main()
    
    # Uncomment to test with a single prediction
    # test_single_prediction()