import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from influxdb import InfluxDBClient
import joblib
import json
from datetime import datetime
import threading
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')

class MQTTClassifier:
    def __init__(self, model_path, scaler_path, influx_config, mqtt_config):
        """Initialize MQTT classifier with model and database connections"""
        # Load trained model
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # MQTT configuration
        self.mqtt_config = mqtt_config
        self.mqtt_topic = mqtt_config.get('topic', 'sound/volume')
        
        # Data buffer for time series windows
        self.window_size = 10
        self.data_buffer = deque(maxlen=50)  # Keep last 50 points
        
        # InfluxDB connection
        self.influx_config = influx_config
        self.setup_influxdb()
        
        # MQTT client
        self.setup_mqtt()
        
        # Statistics
        self.processed_count = 0
        self.last_classification_time = None
        
    def setup_influxdb(self):
        """Initialize InfluxDB connection"""
        try:
            self.influx_client = InfluxDBClient(
                host=self.influx_config['host'],
                port=self.influx_config['port'],
                username=self.influx_config.get('username', ''),
                password=self.influx_config.get('password', ''),
                database=self.influx_config['database']
            )
            
            # Create database if it doesn't exist
            databases = self.influx_client.get_list_database()
            if not any(db['name'] == self.influx_config['database'] for db in databases):
                self.influx_client.create_database(self.influx_config['database'])
            
            self.influx_client.switch_database(self.influx_config['database'])
            print(f"Connected to InfluxDB: {self.influx_config['host']}:{self.influx_config['port']}")
            
        except Exception as e:
            print(f"Error connecting to InfluxDB: {e}")
            raise
    
    def setup_mqtt(self):
        """Initialize MQTT client"""
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        if self.mqtt_config.get('username'):
            self.mqtt_client.username_pw_set(
                self.mqtt_config['username'],
                self.mqtt_config.get('password', '')
            )
    
    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print(f"Connected to MQTT broker at {self.mqtt_config['host']}:{self.mqtt_config['port']}")
            client.subscribe(self.mqtt_topic)
            print(f"Subscribed to topic: {self.mqtt_topic}")
        else:
            print(f"Failed to connect to MQTT broker, return code: {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        print(f"Disconnected from MQTT broker, return code: {rc}")
    
    def on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            # Parse incoming message
            payload = msg.payload.decode('utf-8')
            print(f"Received MQTT message: {payload}")
            
            # Try to parse as JSON
            try:
                data = json.loads(payload)
                volume = data.get('volume')
                peak = data.get('peak')
                timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            except json.JSONDecodeError:
                # If not JSON, try to parse as simple value
                volume = float(payload)
                peak = volume
                timestamp = datetime.utcnow().isoformat()
            
            # Add to buffer with timestamp
            point = {
                'volume': float(volume),
                'peak': float(peak) if peak else float(volume),
                'timestamp': timestamp,
                'received_time': datetime.utcnow()
            }
            
            self.data_buffer.append(point)
            print(f"Added data point: Volume={volume}, Buffer size: {len(self.data_buffer)}")
            
            # Check if we have enough data for classification
            if len(self.data_buffer) >= self.window_size:
                self.process_classification()
            
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def extract_features(self, time_series_data):
        """
        Extract features from time series data (same as training)
        """
        series = np.array(time_series_data)
        # Apply maximum constraint (values <= 0) as in training
        series = np.minimum(series, 0)
        
        # Extract features
        features = [
            np.mean(series),    # Mean of the values
            np.max(series)      # Maximum value
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, time_series_data):
        """
        Predict cluster for time series data
        """
        features = self.extract_features(time_series_data)
        
        # Scale features if needed
        model_type = type(self.model).__name__
        if model_type in ['SVC', 'LogisticRegression']:
            features = self.scaler.transform(features)
        
        cluster = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return cluster, probabilities
    
    def process_classification(self):
        """Process the current buffer for classification"""
        if len(self.data_buffer) < self.window_size:
            return
        
        # Get the most recent window of data
        window_data = list(self.data_buffer)[-self.window_size:]
        volumes = [point['volume'] for point in window_data]
        latest_timestamp = window_data[-1]['timestamp']
        
        try:
            # Perform classification
            cluster, probabilities = self.predict(volumes)
            confidence = np.max(probabilities)
            
            # Prepare result for InfluxDB
            result = {
                "timestamp": latest_timestamp,
                "cluster": int(cluster),
                "probabilities": probabilities.tolist(),
                "confidence": float(confidence),
                "mean_volume": float(np.mean(volumes)),
                "max_volume": float(np.max(volumes)),
                "window_size": self.window_size,
                "data_points": len(volumes)
            }
            
            # Save to InfluxDB
            self.save_to_influxdb(result)
            
            # Update statistics
            self.processed_count += 1
            self.last_classification_time = datetime.utcnow()
            
            print(f"Classification: Cluster {cluster}, Confidence: {confidence:.3f}, "
                  f"Mean: {np.mean(volumes):.2f}, Max: {np.max(volumes):.2f}")
                  
        except Exception as e:
            print(f"Error in classification: {e}")
    
    def save_to_influxdb(self, result):
        """Save classification result to InfluxDB"""
        json_body = [
            {
                "measurement": "sound_classification_real_time",
                "time": str(result["timestamp"]),
                "tags": {
                    "source": "mqtt_realtime",
                    "predicted_cluster": f"cluster_{result['cluster']}",
                    "model_type": type(self.model).__name__
                },
                "fields": {
                    "cluster": result["cluster"],
                    "probability_0": result["probabilities"][0],
                    "probability_1": result["probabilities"][1],
                    "probability_2": result["probabilities"][2],
                    "confidence": result["confidence"],
                    "mean_volume": result["mean_volume"],
                    "max_volume": result["max_volume"],
                    "window_size": result["window_size"],
                    "processed_count": self.processed_count
                }
            }
        ]
        
        try:
            self.influx_client.write_points(json_body)
            print(f"✓ Saved classification to InfluxDB - Cluster {result['cluster']}")
        except Exception as e:
            print(f"Error saving to InfluxDB: {e}")
    
    def start(self):
        """Start the MQTT classifier"""
        try:
            print("Starting MQTT classifier...")
            print(f"Window size: {self.window_size}")
            print(f"Model type: {type(self.model).__name__}")
            
            # Start MQTT client in a separate thread
            mqtt_thread = threading.Thread(target=self._mqtt_loop)
            mqtt_thread.daemon = True
            mqtt_thread.start()
            
            # Start statistics thread
            stats_thread = threading.Thread(target=self._stats_loop)
            stats_thread.daemon = True
            stats_thread.start()
            
            print("MQTT classifier started successfully!")
            print("Press Ctrl+C to stop...")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping MQTT classifier...")
            self.stop()
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.stop()
    
    def _mqtt_loop(self):
        """MQTT network loop"""
        try:
            self.mqtt_client.connect(
                self.mqtt_config['host'],
                self.mqtt_config.get('port', 1883),
                self.mqtt_config.get('keepalive', 60)
            )
            self.mqtt_client.loop_forever()
        except Exception as e:
            print(f"MQTT loop error: {e}")
    
    def _stats_loop(self):
        """Periodic statistics reporting"""
        while True:
            time.sleep(30)  # Report every 30 seconds
            print(f"\n--- Statistics ---")
            print(f"Data buffer size: {len(self.data_buffer)}")
            print(f"Total classifications: {self.processed_count}")
            if self.last_classification_time:
                print(f"Last classification: {self.last_classification_time.strftime('%H:%M:%S')}")
    
    def stop(self):
        """Stop the MQTT classifier"""
        self.mqtt_client.disconnect()
        print("MQTT classifier stopped.")

def main():
    # Configuration
    MODEL_PATH = "best_classification_model_random_forest.pkl"  # Update with your model filename
    SCALER_PATH = "scaler.pkl"
    
    # InfluxDB Configuration
    INFLUX_CONFIG = {
        'host': 'localhost',
        'port': 8086,
        'database': 'iot_data',
        'username': '',  # If required
        'password': ''   # If required
    }
    
    # MQTT Configuration
    MQTT_CONFIG = {
        'host': 'localhost',      # Your MQTT broker host
        'port': 1883,             # Your MQTT broker port
        'topic': 'sound/volume',  # MQTT topic to subscribe to
        'username': 'iot_user',           # If required
        'password': "hosna@8933"            # If required
    }
    
    try:
        # Initialize classifier
        classifier = MQTTClassifier(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            influx_config=INFLUX_CONFIG,
            mqtt_config=MQTT_CONFIG
        )
        
        # Start processing
        classifier.start()
        
    except Exception as e:
        print(f"Failed to start MQTT classifier: {e}")
        import traceback
        traceback.print_exc()

def test_mqtt_publisher():
    """Test function to publish sample MQTT messages"""
    import paho.mqtt.client as mqtt
    import json
    import time
    
    def publish_test_data():
        client = mqtt.Client()
        client.connect('localhost', 1883, 60)
        
        # Simulate different clusters
        cluster_params = {
            0: {'mean': -65, 'std': 25},   # Cluster 0: Low values
            1: {'mean': -40, 'std': 30},    # Cluster 1: Medium values  
            2: {'mean': -15, 'std': 20}     # Cluster 2: High values
        }
        
        print("Publishing test MQTT messages...")
        
        for i in range(50):
            # Rotate through clusters
            cluster = i % 3
            params = cluster_params[cluster]
            
            # Generate sample data
            volume = np.random.normal(params['mean'], params['std'])
            volume = min(volume, 0)  # Apply constraint
            
            message = {
                'volume': float(volume),
                'peak': float(volume * 1.1),
                'timestamp': datetime.utcnow().isoformat(),
                'sequence': i
            }
            
            client.publish('sound/volume', json.dumps(message))
            print(f"Published: {message}")
            time.sleep(1)  # Send every second
        
        client.disconnect()
    
    # Run test publisher in separate thread
    import threading
    publisher_thread = threading.Thread(target=publish_test_data)
    publisher_thread.daemon = True
    publisher_thread.start()

if __name__ == "__main__":
    # Install required packages if not already installed:
    # pip install paho-mqtt influxdb
    
    # Uncomment the next line to test with sample MQTT data
    # test_mqtt_publisher()
    
    # Start the main classifier
    main()