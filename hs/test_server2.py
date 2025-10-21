import paho.mqtt.client as mqtt
import random
import time
from datetime import datetime

# MQTT Configuration
MQTT_BROKER = "5.75.205.93"
MQTT_PORT = 1883
MQTT_TOPIC = "sensors/sound_volume"  # Changed from /json to match your Telegraf config
MQTT_USERNAME = "iot_user"
MQTT_PASSWORD = "hosna@8933"

def send_influx_data():
    # Use Client() instead of Client() to fix deprecation warning
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER, MQTT_PORT)
    
    try:
        message_count = 0
        while True:
            # Generate random sound data
            volume = round(random.uniform(40, 80), 1)
            peak = round(random.uniform(50, 90), 1)
            
            # Create InfluxDB line protocol format
            # Format: measurement,tag=value field=value timestamp
            timestamp = int(time.time() * 1000000000)
            data = f"sound_volume,device=python_mic_01,room=lab volume={volume},peak={peak} {timestamp}"
            
            # Publish
            client.publish(MQTT_TOPIC, data)
            
            message_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] #{message_count}: {volume} dB (peak: {peak} dB)")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.disconnect()

if __name__ == "__main__":
    send_influx_data()