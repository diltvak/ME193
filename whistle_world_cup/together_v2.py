"""Whistle-driven LEGO robot that sings and shouts "object" when it bumps into things.

Combines:
  - tone_driving.py / tone_test.py: whistle F#6 to drive forward, sharper steers
    right, flatter steers left, B6+ reverses; silence stops.
  - color_sensor_test.py: the Color Sensor shines white and calibrates the empty-space
    reflection at startup; a reflection well above that means an object is close,
    and the light blinks red while it is.
  - mqtt_chat.py: each new detection publishes "object" to the ME193 topic (and
    anything else posted there is printed).
  - sensor_song.py: each new detection plays the song on the Mac.
  - whistle_recognition.py: whistling the SpongeBob tune makes the robot spin in
    place for 3 s. The matching runs on a background thread, so whistle driving
    keeps working while it listens.

While an object is in front of the sensor, forward driving is blocked (reverse still
works so you can back away). While the song plays, the mic is ignored so the song
itself can't steer the robot.

A live window shows the microphone waveform with the detected sine overlaid (as in
tone_test.py), the current driving decision and wheel speeds, and a log of MQTT
messages sent and received.

Run: python3 together_v1.py   (Ctrl+C or close the plot window to quit)
"""

import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

import legoeducation as le
import whistle_recognition as wr
from mqttlib import MQTTClient
from tone_test import (
    CHUNK, FFT_SIZE, MAX_FREQ, MIN_FREQ, RATE, SCOPE_MS,
    StabilityGate, calibrate_noise, detect_pitch, fit_sine, freq_to_note, trigger_index,
)

# ---- Configuration: devices ----
MOTOR_CARD_COLOR = le.LEGO_COLOR_PURPLE
MOTOR_CARD_SERIAL = "6235"
SENSOR_CARD_COLOR = None  # None = first Color Sensor found
SENSOR_CARD_SERIAL = None

# ---- Configuration: MQTT ----
TOPIC = "ME193"
OBJECT_MESSAGE = "object"
MQTT_LOG_LINES = 6                # how many recent MQTT messages the window shows

# ---- Configuration: song ----
SONG = Path(__file__).with_name("HungerGames.mp3")
STOP_ON_RELEASE = False           # True = cut the song off when the object moves away
IGNORE_MIC_WHILE_PLAYING = True   # stop listening (and driving) while the song plays,
                                  # so the speakers can't steer the robot

# ---- Configuration: obstacle ----
BLOCK_FORWARD_ON_OBJECT = True    # don't drive forward into a detected object

# ---- Configuration: whistle driving (see tone_driving.py) ----
OCTAVE_SHIFT = 1
FORWARD_NOTE_MIDI = 78 + 12 * OCTAVE_SHIFT  # F#: drive straight
REVERSE_NOTE_MIDI = 83 + 12 * OCTAVE_SHIFT  # B and above: drive backward
STEER_RANGE_SEMITONES = 5
DRIVE_SPEED = 80
REVERSE_SPEED = 80
TURN_SPEED = 5
FORWARD_RANGE_CENTS = 150
SMOOTHING = 0.4
HOLD_TIME_S = 0.3

# ---- Configuration: color sensor (see color_sensor_test.py) ----
SHINE_COLOR = le.LEGO_COLOR_WHITE
BLINK_COLOR = le.LEGO_COLOR_RED
BLINK_PATTERN = le.LIGHT_PATTERN_SHORT_BLINK
INTENSITY = 100
CALIBRATION_S = 2.0
TRIGGER_MARGIN = 10
RELEASE_FRACTION = 0.5
RELEASE_HOLD_S = 0.3
SENSOR_CAL_DELAY_S = 0.05

# ---- Configuration: tune recognition (see whistle_recognition.py) ----
SPIN_SPEED = 75                   # % speed, wheels in opposite directions
SPIN_TIME_S = 3.0


# ---- Whistle driving ----

def note_name(midi):
    """Note name (e.g. 'F#6') for a MIDI number."""
    return freq_to_note(440.0 * 2 ** ((midi - 69) / 12))[0]


def pitch_to_speeds(freq):
    """Map a whistled frequency to (left, right) tank-drive speeds and a label."""
    midi = 69 + 12 * np.log2(freq / 440.0)
    offset = midi - FORWARD_NOTE_MIDI

    if midi >= REVERSE_NOTE_MIDI - 0.5:
        return -REVERSE_SPEED, -REVERSE_SPEED, "REVERSE"
    if abs(offset) > STEER_RANGE_SEMITONES + 0.5:
        return 0.0, 0.0, "out of range"

    if abs(offset) * 100 <= FORWARD_RANGE_CENTS:
        turn = 0.0
    else:
        turn = TURN_SPEED if offset > 0 else -TURN_SPEED

    left = max(-100.0, min(100.0, DRIVE_SPEED + turn))
    right = max(-100.0, min(100.0, DRIVE_SPEED - turn))
    label = "FORWARD" if turn == 0 else ("RIGHT" if turn > 0 else "LEFT")
    return left, right, label


# ---- Color sensor ----

def shine(sensor):
    sensor.light_color(SHINE_COLOR, pattern=le.LIGHT_PATTERN_SOLID, intensity=INTENSITY)


def calibrate_sensor(sensor):
    """Average reflection (%) over CALIBRATION_S seconds."""
    samples = []
    end = time.time() + CALIBRATION_S
    while time.time() < end:
        samples.append(sensor.sensor.reflection)
        time.sleep(SENSOR_CAL_DELAY_S)
    return sum(samples) / len(samples)


class ObjectDetector:
    """Reflection threshold with hysteresis, as in color_sensor_test.py."""

    def __init__(self, baseline):
        self.on_level = min(baseline + TRIGGER_MARGIN, 100)
        self.off_level = baseline + (self.on_level - baseline) * RELEASE_FRACTION
        self.detected = False
        self.low_since = None

    def update(self, level, now):
        """Return 'arrived', 'left', or None for this reading."""
        if not self.detected and level > self.on_level:
            self.detected = True
            self.low_since = None
            return "arrived"
        if self.detected:
            if level >= self.off_level:
                self.low_since = None
            elif self.low_since is None:
                self.low_since = now
            elif now - self.low_since > RELEASE_HOLD_S:
                self.detected = False
                return "left"
        return None


# ---- Song ----

def play_song():
    return subprocess.Popen(["afplay", str(SONG)])


def is_playing(player):
    return player is not None and player.poll() is None


def stop_song(player):
    if is_playing(player):
        player.terminate()
        player.wait()


# ---- Tune recognition ----

class TuneListener:
    """Runs whistle_recognition's tracker and matcher on a background thread.

    The DTW matching can take a noticeable fraction of a second, so the main loop
    only hands audio over with feed() and checks poll() for results; neither blocks.
    """

    def __init__(self, noise_rms):
        if RATE != wr.RATE or CHUNK % wr.CHUNK:
            raise ValueError("Mic chunk must split evenly into whistle_recognition frames")
        self.tracker = wr.PitchTracker(max(wr.MIN_RMS, wr.NOISE_MULTIPLIER * noise_rms))
        self.matcher = wr.TuneMatcher(wr.build_template(wr.REFERENCE))
        self.audio = queue.Queue()
        self.hits = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def feed(self, samples):
        self.audio.put(samples)

    def poll(self):
        """Match cost if the tune was recognized since the last call, else None."""
        try:
            return self.hits.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.audio.put(None)
        self.thread.join(timeout=1.0)

    def _run(self):
        while self.running:
            samples = self.audio.get()
            if samples is None:
                break
            for i in range(0, len(samples), wr.CHUNK):
                pitch = self.tracker.process(samples[i:i + wr.CHUNK])
                cost = self.matcher.update(pitch, time.monotonic())
                if cost is not None:
                    self.hits.put(cost)


def measure_noise_rms(stream):
    """RMS of the room over whistle_recognition's calibration time."""
    n = max(1, int(wr.CALIBRATION_SECONDS * RATE / CHUNK))
    audio = np.concatenate([
        np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        for _ in range(n)
    ])
    return np.sqrt(np.mean(audio ** 2))


# ---- MQTT ----

# The MQTT callback runs on mqttlib's background thread, which must not touch the
# plot, so messages are queued here and drawn by the main loop.
mqtt_log = deque(maxlen=MQTT_LOG_LINES)


def log_mqtt(direction, topic, payload):
    mqtt_log.append(f"{time.strftime('%H:%M:%S')}  {direction}  [{topic}] {payload}")


def on_message(topic, payload):
    log_mqtt("recv <-", topic, payload)
    print(f"\n[{topic}] {payload}")


# ---- Live plot ----

LABEL_COLORS = {
    "FORWARD": "tab:green", "LEFT": "tab:blue", "RIGHT": "tab:blue",
    "REVERSE": "tab:orange", "BLOCKED": "tab:red", "SONG": "tab:purple",
    "SPIN": "tab:pink",
}


def speed_color(speed):
    if speed > 0:
        return "tab:green"
    if speed < 0:
        return "tab:orange"
    return "0.7"


class Dashboard:
    """Scope (mic waveform + detected sine), wheel speeds, and the MQTT log."""

    def __init__(self, mqtt_status):
        self.scope_n = int(RATE * SCOPE_MS / 1000)
        self.t = np.arange(CHUNK) / RATE

        plt.ion()
        self.fig = plt.figure(figsize=(10, 7))
        grid = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 1.4])

        # Oscilloscope, as in tone_test.py.
        self.ax_scope = self.fig.add_subplot(grid[0])
        t_ms = np.arange(self.scope_n) / RATE * 1000
        self.raw_line, = self.ax_scope.plot(t_ms, np.zeros(self.scope_n), color="0.6",
                                            lw=1, label="Microphone")
        self.sine_line, = self.ax_scope.plot(t_ms, np.full(self.scope_n, np.nan),
                                             color="tab:blue", lw=2, label="Detected sine")
        self.ax_scope.set_xlim(0, t_ms[-1])
        self.ax_scope.set_xlabel("Time (ms)")
        self.ax_scope.set_ylabel("Amplitude")
        self.ax_scope.legend(loc="upper right")
        self.ax_scope.grid(alpha=0.3)
        self.scope_title = self.ax_scope.set_title("Listening...")

        # Driving decision: one bar per wheel.
        self.ax_drive = self.fig.add_subplot(grid[1])
        self.bars = self.ax_drive.barh(["Left", "Right"], [0, 0], color="0.7")
        self.ax_drive.invert_yaxis()
        self.ax_drive.set_xlim(-100, 100)
        self.ax_drive.axvline(0, color="0.3", lw=1)
        self.ax_drive.set_xlabel("Wheel speed (%)")
        self.ax_drive.grid(axis="x", alpha=0.3)
        self.drive_title = self.ax_drive.set_title("Drive: STOPPED", fontweight="bold")

        # MQTT log.
        self.ax_mqtt = self.fig.add_subplot(grid[2])
        self.ax_mqtt.axis("off")
        self.ax_mqtt.set_title(f"MQTT  {mqtt_status}", loc="left")
        self.mqtt_text = self.ax_mqtt.text(0, 1, "", va="top", family="monospace",
                                           transform=self.ax_mqtt.transAxes)

        self.fig.tight_layout()
        self.fig.show()

    def is_open(self):
        return plt.fignum_exists(self.fig.number)

    def update(self, samples, freq, heard, label, left, right, level, detected):
        start = trigger_index(samples, CHUNK - self.scope_n)
        view = slice(start, start + self.scope_n)
        self.raw_line.set_ydata(samples[view])
        if freq is None:
            self.sine_line.set_ydata(np.full(self.scope_n, np.nan))
            self.scope_title.set_text(heard.strip())
        else:
            note, cents = freq_to_note(freq)
            self.sine_line.set_ydata(fit_sine(samples, self.t, freq)[view])
            self.scope_title.set_text(f"{note}  {freq:.1f} Hz  ({cents:+.0f} cents)")
        # Auto-scale so quiet whistles are still visible.
        peak = max(np.abs(samples[view]).max(), 0.02)
        self.ax_scope.set_ylim(-1.2 * peak, 1.2 * peak)

        for bar, speed in zip(self.bars, (left, right)):
            bar.set_width(speed)
            bar.set_color(speed_color(speed))
        obj = "OBJECT" if detected else "clear"
        self.drive_title.set_text(f"Drive: {label}   L {left:.0f}%  R {right:.0f}%   "
                                  f"Reflection {level:.0f}% ({obj})")
        self.drive_title.set_color(LABEL_COLORS.get(label, "0.2"))

        self.mqtt_text.set_text("\n".join(mqtt_log) or "(no messages yet)")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    if not SONG.exists():
        print(f"Song not found: {SONG}")
        return

    window = np.hanning(CHUNK)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / RATE)
    band = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)

    doublemotor = le.DoubleMotor()
    sensor = le.ColorSensor()
    mqtt = MQTTClient()
    mqtt_connected = False
    pa = pyaudio.PyAudio()
    stream = None
    player = None
    listener = None

    # try/finally so the motor, light, song and connections are always cleaned up.
    try:
        print("Connecting to Double Motor...")
        doublemotor.connect(card_color=MOTOR_CARD_COLOR, card_serial=MOTOR_CARD_SERIAL)
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return

        print("Connecting to Color Sensor...")
        sensor.connect(card_color=SENSOR_CARD_COLOR, card_serial=SENSOR_CARD_SERIAL)
        if not sensor.connected:
            print("Error connecting to Color Sensor.")
            return

        print(f"Connecting to MQTT broker {mqtt.broker}...")
        try:
            mqtt.connect()
            mqtt.subscribe(TOPIC, on_message)
            mqtt_connected = True
        except OSError as exc:
            print(f"MQTT unavailable ({exc}); continuing without broadcasting.")
        mqtt_status = (f"{mqtt.broker}  topic '{TOPIC}'" if mqtt_connected
                       else "(unavailable, not broadcasting)")

        shine(sensor)
        print("Calibrating Color Sensor: keep objects away from the front of it...")
        detector = ObjectDetector(calibrate_sensor(sensor))
        print(f"Object detected above {detector.on_level:.0f}% reflection, "
              f"released below {detector.off_level:.0f}%.")

        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                         input=True, frames_per_buffer=CHUNK)
        print("Calibrating room noise: stay quiet...")
        noise_profile = calibrate_noise(stream, window, band)
        noise_rms = measure_noise_rms(stream)
        gate = StabilityGate()

        print(f"Loading reference tune {wr.REFERENCE.name}...")
        listener = TuneListener(noise_rms)

        print(f"Whistle {note_name(FORWARD_NOTE_MIDI)} to go forward, higher = right, "
              f"lower = left, {note_name(REVERSE_NOTE_MIDI)} = reverse.")
        print(f"Whistle the {wr.REFERENCE.stem} tune to spin for {SPIN_TIME_S:g} s.")
        print("Ctrl+C or close the plot window to quit.\n")

        dash = Dashboard(mqtt_status)

        smoothed_left = smoothed_right = 0.0
        target_left = target_right = 0.0
        label = "STOPPED"
        last_heard = 0.0
        spin_until = 0.0

        while dash.is_open():
            # Always read the mic so its buffer never backs up, even when ignoring it.
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)
            now = time.time()

            # ---- Tune recognition (runs on its own thread) ----
            # While the song plays it gets silence instead, so the speakers can't
            # trigger it but its timeline keeps moving.
            song_on = IGNORE_MIC_WHILE_PLAYING and is_playing(player)
            listener.feed(np.zeros_like(samples) if song_on else samples)
            cost = listener.poll()
            if cost is not None:
                print(f"\n*** {wr.REFERENCE.stem} whistle recognized (cost {cost:.2f}): "
                      f"spinning for {SPIN_TIME_S:g} s ***")
                spin_until = now + SPIN_TIME_S
            spinning = now < spin_until

            # ---- Object detection ----
            level = sensor.sensor.reflection
            event = detector.update(level, now)
            if event == "arrived":
                sensor.light_color(BLINK_COLOR, pattern=BLINK_PATTERN, intensity=INTENSITY)
                if mqtt_connected:
                    mqtt.publish(TOPIC, OBJECT_MESSAGE)
                    log_mqtt("sent ->", TOPIC, OBJECT_MESSAGE)
                if not is_playing(player):
                    player = play_song()
            elif event == "left":
                shine(sensor)
                if STOP_ON_RELEASE:
                    stop_song(player)

            # ---- Whistle -> drive command ----
            freq = None
            if song_on:
                gate.update(None)
                target_left = target_right = 0.0
                label = "SONG"
                heard = "   (song playing, mic ignored)"
            else:
                raw_freq, snr = detect_pitch(samples, window, freqs, band, noise_profile)
                freq = gate.update(raw_freq)
                if freq is not None:
                    target_left, target_right, label = pitch_to_speeds(freq)
                    note, cents = freq_to_note(freq)
                    heard = f"{note:>4} {freq:7.1f} Hz {cents:+4.0f}c SNR {snr:3.0f}dB"
                    last_heard = now
                elif now - last_heard > HOLD_TIME_S:
                    target_left = target_right = 0.0
                    label = "STOPPED"
                    heard = "   (no confident whistle)"
                else:
                    heard = "   (holding)"

            if BLOCK_FORWARD_ON_OBJECT and detector.detected and target_left + target_right > 0:
                target_left = target_right = 0.0
                label = "BLOCKED"

            # A recognized tune overrides whistle driving: spin in place, full effect
            # right away (no smoothing ramp).
            if spinning:
                target_left, target_right = SPIN_SPEED, -SPIN_SPEED
                smoothed_left, smoothed_right = target_left, target_right
                label = "SPIN"
            elif target_left == 0 and target_right == 0:
                smoothed_left = smoothed_right = 0.0
            else:
                smoothed_left += (target_left - smoothed_left) * SMOOTHING
                smoothed_right += (target_right - smoothed_right) * SMOOTHING

            doublemotor.movement_move_tank(
                speed_left=int(smoothed_left),
                speed_right=int(smoothed_right),
                blocking=False,
            )

            obj = "OBJECT" if detector.detected else "clear"
            line = (f"{heard:<38} {label:<13} L: {smoothed_left:4.0f}%  R: {smoothed_right:4.0f}%"
                    f"  Refl: {level:3.0f}% {obj}")
            print(f"\r{line:<105}", end="", flush=True)

            dash.update(samples, freq, heard, label, smoothed_left, smoothed_right,
                        level, detector.detected)
        print("\nPlot closed.")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("Cleaning up...")
        plt.close("all")
        if listener is not None:
            listener.stop()
        stop_song(player)
        if doublemotor.connected:
            try:
                doublemotor.movement_stop()
            except Exception as exc:
                print(f"Error stopping motor: {exc}")
            doublemotor.disconnect()
        if sensor.connected:
            try:
                sensor.light_color(le.LEGO_COLOR_NOCOLOR, intensity=0)
            except Exception as exc:
                print(f"Error turning off light: {exc}")
            sensor.disconnect()
        if mqtt_connected:
            mqtt.disconnect()
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
