#!/usr/bin/env python3
"""
Debug script to see what's actually in the database
"""

from influxdb import InfluxDBClient

def debug_influxdb():
    SERVER_IP = "5.75.205.93"
    INFLUXDB_PORT = 8086
    DATABASE_NAME = "iot_data"
    
    print(f"🔍 Debugging InfluxDB at {SERVER_IP}...")
    
    try:
        client = InfluxDBClient(host=SERVER_IP, port=INFLUXDB_PORT, database=DATABASE_NAME)
        
        # Check all measurements
        print("\n📋 ALL MEASUREMENTS:")
        measurements = client.get_list_measurements()
        if measurements:
            for measurement in measurements:
                print(f"  - {measurement['name']}")
        else:
            print("  No measurements found!")
        
        # Check all databases
        print("\n🗄️ ALL DATABASES:")
        databases = client.get_list_database()
        for db in databases:
            print(f"  - {db['name']}")
        
        # Show any data in the database
        print("\n🔎 ANY DATA IN IOT_DATA DATABASE:")
        result = client.query('SHOW SERIES ON iot_data')
        if result:
            for point in result.get_points():
                print(f"  {point}")
        else:
            print("  No series found in iot_data database!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_influxdb()