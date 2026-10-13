"""Publishes the record number of continuous hits to ME193/Rogers/<name> over MQTT.

The score is a float (e.g. 7.0) and is published retained, in real time, each time the record
improves. Connection problems never stop the game: publishing just becomes a no-op.
"""

import threading

from powerup_pong.config import MQTT_BROKER, MQTT_TOPIC_FMT
from powerup_pong.mqttlib import MQTTClient


class Scoreboard:
    def __init__(self, name, broker=MQTT_BROKER, enabled=True):
        self.topic = MQTT_TOPIC_FMT.format(name=name)
        self.record = 0.0
        self._client = None
        self._broker = broker
        if enabled:   # connecting can take seconds, so keep it off the camera loop
            threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        try:
            client = MQTTClient(broker=self._broker)
            client.connect()
            self._client = client
            print(f"Scoreboard: publishing to {self._broker} on {self.topic}")
            self._publish()
        except Exception as err:
            print(f"Scoreboard disabled: {err}")

    @property
    def online(self):
        return self._client is not None

    def report(self, rally):
        """Call with the current continuous-hit count; publishes only when it beats the record."""
        if rally > self.record:
            self.record = float(rally)
            self._publish()

    def _publish(self):
        client = self._client
        if client:
            try:
                client.publish(self.topic, str(self.record), retain=True)
            except Exception as err:
                print(f"Scoreboard publish failed: {err}")

    def close(self):
        if self._client:
            self._client.disconnect()
