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

def generate_clustered_value():
    # Choose a random cluster
    cluster = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
    
    if cluster == 1:
        # First cluster: mean -65, std 15
        value = random.gauss(-65, 15)
    elif cluster == 2:
        # Second cluster: mean -40, std 20
        value = random.gauss(-40, 20)
    else:
        # Third cluster: mean -15, std 10
        value = random.gauss(-15, 10)
    
    # Ensure value doesn't exceed 0 (maximum)
    value = min(value, 0)
    
    return round(value, 1), cluster

def send_influx_data():
    # Use Client() instead of Client() to fix deprecation warning
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER, MQTT_PORT)
    
    try:
        message_count = 0
        while True:
            # Generate clustered sound data
            volume, cluster = generate_clustered_value()
            peak = volume+10+ random.random()*10#, _ = generate_clustered_value()
            
            # Create InfluxDB line protocol format
            # Format: measurement,tag=value field=value timestamp
            timestamp = int(time.time() * 1000000000)
            data = f"sound_volume,device=python_mic_01,room=lab volume={volume},peak={peak} {timestamp}"
            
            # Publish
            client.publish(MQTT_TOPIC, data)
            
            message_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] #{message_count}: Cluster {cluster}, Volume: {volume} dB (peak: {peak} dB)")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.disconnect()

if __name__ == "__main__":
    send_influx_data()