#!/usr/bin/env python3
import yaml
import paho.mqtt.client as mqtt
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
import threading
import logging
from datetime import datetime

class RealTimeAIProcessor:
    def __init__(self, config_path="ai_config.yaml"):
        self.load_config(config_path)
        self.setup_logging()
        self.setup_ai_models()
        self.data_buffers = {}
        self.setup_mqtt_client()
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_ai_models(self):
        self.models = {}
        for sensor in self.config['sensor_measurements']:
            if sensor['processing']['anomaly_detection']:
                self.models[sensor['name']] = IsolationForest(
                    contamination=self.config['ai_models']['anomaly_detection']['contamination'],
                    random_state=self.config['ai_models']['anomaly_detection']['random_state']
                )
            # Initialize data buffer for each sensor
            self.data_buffers[sensor['name']] = deque(
                maxlen=sensor['processing']['window_size']
            )
            
    def setup_mqtt_client(self):
        self.client = mqtt.Client()
        self.client.tls_set(
            ca_certs=self.config['mqtt_config']['ca_cert']
        )
        self.client.username_pw_set(
            self.config['mqtt_config']['username'],
            self.config['mqtt_config']['password']
        )
        
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc):
        self.logger.info(f"Connected to MQTT broker with result code {rc}")
        # Subscribe to all configured sensor topics
        for sensor in self.config['sensor_measurements']:
            client.subscribe(sensor['mqtt_topic'])
            self.logger.info(f"Subscribed to {sensor['mqtt_topic']}")
            
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.process_realtime_data(msg.topic, payload)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            
    def process_realtime_data(self, topic, payload):
        # Find which sensor this topic belongs to
        for sensor in self.config['sensor_measurements']:
            if self.topic_matches(topic, sensor['mqtt_topic']):
                value = payload.get(sensor['field'])
                if value is not None:
                    self.analyze_sensor_data(sensor['name'], value, payload)
                break
                
    def topic_matches(self, topic, pattern):
        # Simple wildcard matching (replace + with .* and # with .*)
        import re
        regex_pattern = pattern.replace('+', '[^/]+').replace('#', '.+')
        return re.match(regex_pattern, topic) is not None
        
    def analyze_sensor_data(self, sensor_name, value, full_payload):
        # Add to buffer
        buffer = self.data_buffers[sensor_name]
        buffer.append(value)
        
        # Check threshold
        sensor_config = next(s for s in self.config['sensor_measurements'] if s['name'] == sensor_name)
        if value > sensor_config['processing']['threshold']:
            self.logger.warning(f"Threshold exceeded for {sensor_name}: {value}")
            
        # Anomaly detection when buffer is full
        if len(buffer) >= sensor_config['processing']['window_size']:
            if sensor_config['processing']['anomaly_detection']:
                self.detect_anomalies(sensor_name, list(buffer))
                
        # Log processing result
        self.logger.info(f"Processed {sensor_name}: {value}")
        
    def detect_anomalies(self, sensor_name, data):
        model = self.models[sensor_name]
        # Reshape for sklearn
        data_array = np.array(data).reshape(-1, 1)
        
        # Fit and predict (in real scenario, you'd pre-train or update periodically)
        predictions = model.fit_predict(data_array)
        anomalies = np.where(predictions == -1)[0]
        
        if len(anomalies) > 0:
            self.logger.warning(f"Anomalies detected in {sensor_name} at positions: {anomalies}")
            
    def run(self):
        self.client.connect(
            self.config['mqtt_config']['host'],
            self.config['mqtt_config']['port'],
            60
        )
        self.logger.info("Starting MQTT AI processor...")
        self.client.loop_forever()

if __name__ == "__main__":
    processor = RealTimeAIProcessor()
    processor.run()