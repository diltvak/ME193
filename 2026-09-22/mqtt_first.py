import time

from mqttlib import MQTTClient

TOPIC = "ME193"
MESSAGE = "David is trying to reach the portal..."

def on_message(topic, payload):
    print(f"Got it back: [{topic}] {payload}")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    time.sleep(1)  # give the subscription time to reach the broker

    client.publish(TOPIC, "hello world")
    print(f"Published '{MESSAGE}' to '{TOPIC}' on test.mosquitto.org")

    time.sleep(1)  # give the message time to come back before disconnecting