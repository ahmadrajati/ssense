#!/usr/bin/env python3
import yaml
import argparse
from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import logging

class HistoricalAIProcessor:
    def __init__(self, config_path="ai_config.yaml"):
        self.load_config(config_path)
        self.setup_logging()
        self.setup_influx_client()
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_influx_client(self):
        influx_config = self.config['influx_config']
        self.client = InfluxDBClient(
            host=influx_config['host'],
            port=influx_config['port'],
            username=influx_config['username'],
            password=influx_config['password'],
            database=influx_config['database']
        )
        
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Historical AI Data Processor')
        parser.add_argument('--start_time', required=True, help='Start time (YYYY-MM-DD HH:MM:SS)')
        parser.add_argument('--end_time', required=True, help='End time (YYYY-MM-DD HH:MM:SS)')
        parser.add_argument('--sensor', help='Specific sensor to analyze (optional)')
        return parser.parse_args()
        
    def fetch_data(self, start_time, end_time, sensor_name=None):
        sensors_to_process = self.config['sensor_measurements']
        if sensor_name:
            sensors_to_process = [s for s in sensors_to_process if s['name'] == sensor_name]
            
        all_results = {}
        
        for sensor in sensors_to_process:
            query = f'''
            SELECT "{sensor['field']}" FROM "{sensor['influx_measurement']}" 
            WHERE time >= '{start_time}' AND time <= '{end_time}'
            '''
            
            self.logger.info(f"Querying data for {sensor['name']}")
            result = self.client.query(query)
            
            points = list(result.get_points())
            if points:
                df = pd.DataFrame(points)
                df['time'] = pd.to_datetime(df['time'])
                all_results[sensor['name']] = df
                
        return all_results
        
    def analyze_historical_data(self, data_dict):
        analysis_results = {}
        
        for sensor_name, df in data_dict.items():
            sensor_config = next(s for s in self.config['sensor_measurements'] if s['name'] == sensor_name)
            field = sensor_config['field']
            
            # Basic statistics
            values = df[field].values
            stats = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
            
            # Trend analysis
            if len(values) > 1:
                trend = self.calculate_trend(values)
                stats['trend'] = trend
                
            # Anomaly detection
            if sensor_config['processing']['anomaly_detection']:
                anomalies = self.detect_historical_anomalies(values, sensor_config)
                stats['anomaly_count'] = len(anomalies)
                stats['anomaly_indices'] = anomalies
                
            analysis_results[sensor_name] = stats
            
        return analysis_results
        
    def calculate_trend(self, values):
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
        
    def detect_historical_anomalies(self, values, sensor_config):
        if len(values) < sensor_config['processing']['window_size']:
            return []
            
        model = IsolationForest(
            contamination=self.config['ai_models']['anomaly_detection']['contamination'],
            random_state=self.config['ai_models']['anomaly_detection']['random_state']
        )
        
        values_2d = values.reshape(-1, 1)
        predictions = model.fit_predict(values_2d)
        anomalies = np.where(predictions == -1)[0]
        
        return anomalies.tolist()
        
    def generate_report(self, analysis_results):
        self.logger.info("=== HISTORICAL DATA ANALYSIS REPORT ===")
        for sensor_name, results in analysis_results.items():
            self.logger.info(f"\nSensor: {sensor_name}")
            for key, value in results.items():
                if key != 'anomaly_indices':  # Don't log all indices
                    self.logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
                    
    def run(self):
        args = self.parse_arguments()
        
        self.logger.info(f"Processing historical data from {args.start_time} to {args.end_time}")
        
        # Fetch data
        data = self.fetch_data(args.start_time, args.end_time, args.sensor)
        
        if not data:
            self.logger.warning("No data found for the specified time range")
            return
            
        # Analyze data
        results = self.analyze_historical_data(data)
        
        # Generate report
        self.generate_report(results)

if __name__ == "__main__":
    processor = HistoricalAIProcessor()
    processor.run()