from mqttlib import MQTTClient

TOPIC = "ME193"


def on_message(topic, payload):
    print(f"\n[{topic}] {payload}\n> ", end="")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    print(f"Listening on '{TOPIC}'. Type a message and press Enter to publish (Ctrl+C to quit).")

    try:
        while True:
            message = input("> ")
            if message:
                client.publish(TOPIC, message)
    except (KeyboardInterrupt, EOFError):
        print("\nDisconnecting...")
