"""Play the Hunger Games whistle on the Mac when the Color Sensor sees an object.

Uses the same detection as color_sensor_test.py: on startup the script measures
how much light bounces back with nothing in front of the sensor (keep objects
away), then treats a reflection well above that baseline as "object detected".

Each time an object arrives the song plays once (through macOS's built-in
`afplay`). It won't restart while it's already playing, and it re-arms once the
object moves away, so the next object plays it again.

Run: python3 sensor_song.py   (Ctrl+C to quit)
"""

import subprocess
import time
from pathlib import Path

import legoeducation as le

# ---- Configuration ----
CARD_COLOR = None         # e.g. le.LEGO_COLOR_PURPLE; None = first Color Sensor found
CARD_SERIAL = None        # e.g. "6235"; None = any serial

SONG = Path(__file__).with_name("Happy_Short.mp3")
STOP_ON_RELEASE = False   # True = cut the song off when the object moves away

SHINE_COLOR = le.LEGO_COLOR_WHITE              # always-on light while nothing is near
BLINK_COLOR = le.LEGO_COLOR_RED                # blinks this while an object is near
BLINK_PATTERN = le.LIGHT_PATTERN_SHORT_BLINK
INTENSITY = 100                                # light brightness (%)

CALIBRATION_S = 2.0       # how long to measure the empty-space reflection at startup
TRIGGER_MARGIN = 10       # detect when reflection (%) rises this far above the baseline
RELEASE_FRACTION = 0.5    # release once reflection falls below this fraction of the
                          # way from baseline to trigger (hysteresis stops flicker)
RELEASE_HOLD_S = 0.3      # ...and has stayed there this long
LOOP_DELAY_S = 0.05


def shine(sensor):
    sensor.light_color(SHINE_COLOR, pattern=le.LIGHT_PATTERN_SOLID, intensity=INTENSITY)


def calibrate(sensor):
    """Average reflection (%) over CALIBRATION_S seconds."""
    samples = []
    end = time.time() + CALIBRATION_S
    while time.time() < end:
        samples.append(sensor.sensor.reflection)
        time.sleep(LOOP_DELAY_S)
    return sum(samples) / len(samples)


def play_song():
    """Start the song in the background and return its process."""
    return subprocess.Popen(["afplay", str(SONG)])


def is_playing(player):
    return player is not None and player.poll() is None


def stop_song(player):
    if is_playing(player):
        player.terminate()
        player.wait()


def main():
    if not SONG.exists():
        print(f"Song not found: {SONG}")
        return

    sensor = le.ColorSensor()
    player = None

    # try/finally so the light and the song are always stopped, however the script ends.
    try:
        print("Connecting to Color Sensor...")
        sensor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
        if not sensor.connected:
            print("Error connecting to Color Sensor.")
            return

        shine(sensor)

        print("Calibrating: keep objects away from the front of the sensor...")
        baseline = calibrate(sensor)
        on_level = min(baseline + TRIGGER_MARGIN, 100)
        off_level = baseline + (on_level - baseline) * RELEASE_FRACTION
        print(f"Baseline reflection {baseline:.0f}%, detect above {on_level:.0f}%, "
              f"release below {off_level:.0f}%.")
        print("Bring an object close to the sensor. Ctrl+C to quit.\n")

        detected = False
        low_since = None
        while True:
            level = sensor.sensor.reflection
            now = time.time()

            if not detected and level > on_level:
                detected = True
                low_since = None
                sensor.light_color(BLINK_COLOR, pattern=BLINK_PATTERN, intensity=INTENSITY)
                if not is_playing(player):
                    player = play_song()
            elif detected:
                if level >= off_level:
                    low_since = None
                elif low_since is None:
                    low_since = now
                elif now - low_since > RELEASE_HOLD_S:
                    detected = False
                    shine(sensor)
                    if STOP_ON_RELEASE:
                        stop_song(player)

            if detected:
                state = "OBJECT DETECTED"
            elif is_playing(player):
                state = "playing"
            else:
                state = "waiting"
            print(f"\rReflection: {level:3.0f}%   {state:<16}", end="", flush=True)
            time.sleep(LOOP_DELAY_S)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("Cleaning up...")
        stop_song(player)
        if sensor.connected:
            try:
                sensor.light_color(le.LEGO_COLOR_NOCOLOR, intensity=0)
            except Exception as exc:
                print(f"Error turning off light: {exc}")
            sensor.disconnect()


if __name__ == "__main__":
    main()
