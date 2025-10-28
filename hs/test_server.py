
import paho.mqtt.client as mqtt
import random
import time
import json
from datetime import datetime

# MQTT Configuration
MQTT_BROKER = "5.75.205.93"
MQTT_PORT = 1883
MQTT_TOPIC = "sensors/sound/json"
MQTT_USERNAME = "iot_user"
MQTT_PASSWORD = "hosna@8933"

def send_json_data():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER, MQTT_PORT)
    
    try:
        message_count = 0
        while True:
            # Create JSON data
            data = {
                "measurement": "sound_volume",
                "tags": {
                    "device": "python_mic_01",
                    "room": "lab"
                },
                "fields": {
                    "volume": round(random.uniform(-80, 0), 1),
                    "peak": round(random.uniform(-50, 0), 1)
                },
                "timestamp": int(time.time() * 1000000000)
            }
            
            # Convert to JSON string
            json_data = json.dumps(data)
            
            # Publish
            client.publish(MQTT_TOPIC, json_data)
            
            message_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] #{message_count}: {data['fields']['volume']} dB")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.disconnect()

if __name__ == "__main__":
    send_json_data()


"""client = mqtt.Client()
client.username_pw_set(username, password)
client.connect(broker)

try:
    for i in range(10):
        temp = round(random.uniform(20.0, 30.0), 2)
        timestamp = int(time.time() * 1000000000)  # nanoseconds
        message = f"temperature,device=python_sensor value={temp} {timestamp}"
        
        client.publish(topic, message)
        print(f"Sent: {message}")
        time.sleep(1)
        
finally:
    client.disconnect()"""