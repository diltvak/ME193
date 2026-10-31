"""
Listen on the MQTT topic ME193/Sumo and print every message to the terminal.

Usage: python listen_sumo.py
Press Ctrl+C to stop.
"""

import time

from mqttlib import MQTTClient

TOPIC = "ME193/Sumo"


def on_message(topic, payload):
    print(f"[{time.strftime('%H:%M:%S')}] {topic}: {payload}", flush=True)


def main():
    with MQTTClient() as client:
        client.subscribe(TOPIC, on_message)
        print(f"Listening on {TOPIC} (Ctrl+C to quit)...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
