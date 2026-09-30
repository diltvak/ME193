"""Whistle-driven LEGO goalie that plays a song when told it blocked or conceded a shot.

Combines:
  - tone_driving.py / tone_test.py: whistle F#6 to drive forward, sharper steers
    right, flatter steers left, B6+ reverses; silence stops.
  - mqtt_chat.py: listens on the ME193 topic. "blocked" plays Happy_Short.mp3 and
    "scored" plays HungerGames.mp3 (anything else posted there is just logged).

While a song plays, the mic is ignored so the song itself can't steer the robot. A new
"blocked"/"scored" message cuts off whatever song is playing and starts the new one.

A live window shows the microphone waveform with the detected sine overlaid (as in
tone_test.py), the current driving decision and wheel speeds, and a log of MQTT
messages received.

Run: python3 goalie.py   (Ctrl+C or close the plot window to quit)
"""

import queue
import subprocess
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

import legoeducation as le
from mqttlib import MQTTClient
from tone_test import (
    CHUNK, FFT_SIZE, MAX_FREQ, MIN_FREQ, RATE, SCOPE_MS,
    StabilityGate, calibrate_noise, detect_pitch, fit_sine, freq_to_note, trigger_index,
)

# ---- Configuration: devices ----
MOTOR_CARD_COLOR = le.LEGO_COLOR_PURPLE
MOTOR_CARD_SERIAL = "6235"

# ---- Configuration: MQTT ----
TOPIC = "ME193"
MQTT_LOG_LINES = 6                # how many recent MQTT messages the window shows

# ---- Configuration: songs ----
HERE = Path(__file__).parent
SONGS = {                         # MQTT message (case-insensitive) -> song to play
    "blocked": HERE / "Happy_Short.mp3",
    "scored": HERE / "HungerGames.mp3",
}
IGNORE_MIC_WHILE_PLAYING = True   # stop listening (and driving) while a song plays,
                                  # so the speakers can't steer the robot

# ---- Configuration: whistle driving (see tone_driving.py) ----
OCTAVE_SHIFT = 1
FORWARD_NOTE_MIDI = 78 + 12 * OCTAVE_SHIFT  # F#: drive straight
REVERSE_NOTE_MIDI = 83 + 12 * OCTAVE_SHIFT  # B and above: drive backward
STEER_RANGE_SEMITONES = 5
DRIVE_SPEED = 80
REVERSE_SPEED = 45
TURN_SPEED = 5
FORWARD_RANGE_CENTS = 150
SMOOTHING = 0.4
HOLD_TIME_S = 0.3


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


# ---- Songs ----

def play_song(song):
    return subprocess.Popen(["afplay", str(song)])


def is_playing(player):
    return player is not None and player.poll() is None


def stop_song(player):
    if is_playing(player):
        player.terminate()
        player.wait()


# ---- MQTT ----

# The MQTT callback runs on mqttlib's background thread, which must not touch the
# plot or start songs, so messages are queued here and handled by the main loop.
mqtt_log = deque(maxlen=MQTT_LOG_LINES)
song_requests = queue.Queue()


def log_mqtt(direction, topic, payload):
    mqtt_log.append(f"{time.strftime('%H:%M:%S')}  {direction}  [{topic}] {payload}")


def on_message(topic, payload):
    log_mqtt("recv <-", topic, payload)
    print(f"\n[{topic}] {payload}")
    command = payload.strip().lower()
    if command in SONGS:
        song_requests.put(command)


# ---- Live plot ----

LABEL_COLORS = {
    "FORWARD": "tab:green", "LEFT": "tab:blue", "RIGHT": "tab:blue",
    "REVERSE": "tab:orange", "SONG": "tab:purple",
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

    def update(self, samples, freq, heard, label, left, right, song):
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
        playing = f"   Playing: {song}" if song else ""
        self.drive_title.set_text(f"Drive: {label}   L {left:.0f}%  R {right:.0f}%{playing}")
        self.drive_title.set_color(LABEL_COLORS.get(label, "0.2"))

        self.mqtt_text.set_text("\n".join(mqtt_log) or "(no messages yet)")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    missing = [str(song) for song in SONGS.values() if not song.exists()]
    if missing:
        print("Song(s) not found: " + ", ".join(missing))
        return

    window = np.hanning(CHUNK)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / RATE)
    band = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)

    doublemotor = le.DoubleMotor()
    mqtt = MQTTClient()
    mqtt_connected = False
    pa = pyaudio.PyAudio()
    stream = None
    player = None
    song_name = None

    # try/finally so the motor, song and connections are always cleaned up.
    try:
        print("Connecting to Double Motor...")
        doublemotor.connect(card_color=MOTOR_CARD_COLOR, card_serial=MOTOR_CARD_SERIAL)
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return

        print(f"Connecting to MQTT broker {mqtt.broker}...")
        try:
            mqtt.connect()
            mqtt.subscribe(TOPIC, on_message)
            mqtt_connected = True
        except OSError as exc:
            print(f"MQTT unavailable ({exc}); continuing without song triggers.")
        mqtt_status = (f"{mqtt.broker}  topic '{TOPIC}'  (listening for "
                       + ", ".join(f"'{m}'" for m in SONGS) + ")"
                       if mqtt_connected else "(unavailable, songs won't trigger)")

        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                         input=True, frames_per_buffer=CHUNK)
        print("Calibrating room noise: stay quiet...")
        noise_profile = calibrate_noise(stream, window, band)
        gate = StabilityGate()

        print(f"Whistle {note_name(FORWARD_NOTE_MIDI)} to go forward, higher = right, "
              f"lower = left, {note_name(REVERSE_NOTE_MIDI)} = reverse.")
        print("Ctrl+C or close the plot window to quit.\n")

        dash = Dashboard(mqtt_status)

        smoothed_left = smoothed_right = 0.0
        target_left = target_right = 0.0
        label = "STOPPED"
        last_heard = 0.0

        while dash.is_open():
            # Always read the mic so its buffer never backs up, even when ignoring it.
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)
            now = time.time()

            # ---- MQTT song triggers (latest message wins) ----
            while not song_requests.empty():
                command = song_requests.get_nowait()
                stop_song(player)
                player = play_song(SONGS[command])
                song_name = SONGS[command].name
            if not is_playing(player):
                song_name = None

            # ---- Whistle -> drive command ----
            freq = None
            if IGNORE_MIC_WHILE_PLAYING and is_playing(player):
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

            if target_left == 0 and target_right == 0:
                smoothed_left = smoothed_right = 0.0
            else:
                smoothed_left += (target_left - smoothed_left) * SMOOTHING
                smoothed_right += (target_right - smoothed_right) * SMOOTHING

            doublemotor.movement_move_tank(
                speed_left=int(smoothed_left),
                speed_right=int(smoothed_right),
                blocking=False,
            )

            line = (f"{heard:<38} {label:<13} L: {smoothed_left:4.0f}%  R: {smoothed_right:4.0f}%"
                    f"  {song_name or ''}")
            print(f"\r{line:<105}", end="", flush=True)

            dash.update(samples, freq, heard, label, smoothed_left, smoothed_right, song_name)
        print("\nPlot closed.")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("Cleaning up...")
        plt.close("all")
        stop_song(player)
        if doublemotor.connected:
            try:
                doublemotor.movement_stop()
            except Exception as exc:
                print(f"Error stopping motor: {exc}")
            doublemotor.disconnect()
        if mqtt_connected:
            mqtt.disconnect()
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
