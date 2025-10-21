#!/usr/bin/env python3
"""
Script to check InfluxDB data from OUTSIDE the server
"""

from influxdb import InfluxDBClient
import requests

def check_influxdb_remote():
    # Configuration - UPDATE THESE WITH YOUR SERVER DETAILS!
    SERVER_IP = "5.75.205.93"  # e.g., "192.168.1.100" or "53.31.123.12"
    INFLUXDB_PORT = 8086
    DATABASE_NAME = "iot_data"
    # Add username/password if you enabled auth
    USERNAME = ""
    PASSWORD = ""
    
    print(f"🔍 Checking InfluxDB at {SERVER_IP}:{INFLUXDB_PORT}...")
    
    try:
        # Create InfluxDB client
        client = InfluxDBClient(
            host=SERVER_IP,
            port=INFLUXDB_PORT,
            username=USERNAME,
            password=PASSWORD,
            database=DATABASE_NAME
        )
        
        # Check if we can connect
        print("✅ Connected to InfluxDB successfully!")
        
        # Show all measurements
        print("\n📋 Measurements in database:")
        measurements = client.get_list_measurements()
        for measurement in measurements:
            print(f"  - {measurement['name']}")
        
        # Check sound_volume data
        print(f"\n📊 Latest sound_volume data:")
        query = 'SELECT * FROM sound_volume ORDER BY time DESC LIMIT 5'
        result = client.query(query)
        
        if result:
            for point in result.get_points():
                print(f"  Time: {point['time']}")
                print(f"  Device: {point.get('device', 'N/A')}")
                print(f"  Volume: {point.get('value', point.get('volume', 'N/A'))} dB")
                print(f"  Room: {point.get('room', 'N/A')}")
                print("  ---")
        else:
            print("  No data found in sound_volume measurement")
            
        # Show some stats
        print(f"\n📈 Data statistics:")
        count_query = 'SELECT COUNT("value") FROM sound_volume'
        count_result = client.query(count_query)
        for point in count_result.get_points():
            print(f"  Total data points: {point['count']}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to InfluxDB at {SERVER_IP}:{INFLUXDB_PORT}")
        print("   Check if:")
        print("   1. Server IP is correct")
        print("   2. InfluxDB is running on the server")
        print("   3. Port 8086 is open in firewall")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_influxdb_remote()