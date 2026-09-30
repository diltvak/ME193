"""Blink the LEGO Education Color Sensor's light when an object comes near.

The sensor's light shines solid SHINE_COLOR at full brightness the whole time.
On startup the script measures how much light bounces back with nothing in
front of the sensor (keep objects away). When an object gets close enough that
the reflected light rises well above that baseline, the sensor's light blinks
BLINK_COLOR. When the object moves away, it goes back to shining solid.

Run: python3 color_sensor_test.py   (Ctrl+C to quit)
"""

import time

import legoeducation as le

# ---- Configuration ----
CARD_COLOR = None         # e.g. le.LEGO_COLOR_PURPLE; None = first Color Sensor found
CARD_SERIAL = None        # e.g. "6235"; None = any serial

SHINE_COLOR = le.LEGO_COLOR_WHITE              # always-on light while nothing is near
BLINK_COLOR = le.LEGO_COLOR_RED                # blinks this when an object is near
BLINK_PATTERN = le.LIGHT_PATTERN_SHORT_BLINK   # or LONG_BLINK / DOUBLE_BLINK / PULSE
INTENSITY = 100                                # light brightness (%)

CALIBRATION_S = 2.0       # how long to measure the empty-space reflection at startup
TRIGGER_MARGIN = 10       # blink when reflection (%) rises this far above the baseline
RELEASE_FRACTION = 0.5    # stop once reflection falls back below this fraction of the
                          # way from baseline to trigger (hysteresis stops flicker)
RELEASE_HOLD_S = 0.3      # ...and has stayed there this long, so the blink's own dark
                          # moments don't end it early
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


def main():
    sensor = le.ColorSensor()

    # try/finally so the light is always turned off, however the script ends.
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
        print(f"Baseline reflection {baseline:.0f}%, blink above {on_level:.0f}%, "
              f"stop below {off_level:.0f}%.")
        print("Bring an object close to the sensor. Ctrl+C to quit.\n")

        blinking = False
        low_since = None
        while True:
            level = sensor.sensor.reflection
            now = time.time()

            if not blinking and level > on_level:
                sensor.light_color(BLINK_COLOR, pattern=BLINK_PATTERN, intensity=INTENSITY)
                blinking = True
                low_since = None
            elif blinking:
                if level >= off_level:
                    low_since = None
                elif low_since is None:
                    low_since = now
                elif now - low_since > RELEASE_HOLD_S:
                    shine(sensor)
                    blinking = False

            state = "OBJECT DETECTED - blinking" if blinking else "shining"
            print(f"\rReflection: {level:3.0f}%   {state:<27}", end="", flush=True)
            time.sleep(LOOP_DELAY_S)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("Cleaning up...")
        if sensor.connected:
            try:
                sensor.light_color(le.LEGO_COLOR_NOCOLOR, intensity=0)
            except Exception as exc:
                print(f"Error turning off light: {exc}")
            sensor.disconnect()


if __name__ == "__main__":
    main()
