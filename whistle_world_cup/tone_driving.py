"""Steer a LEGO Education Double Motor robot by whistling.

Whistle the forward note (F#6 by default) to drive straight forward. Whistling
sharper (higher) than it steers right, flatter (lower) steers left, always at
the same TURN_SPEED. Whistling the reverse note
(B6 by default) or above backs up. OCTAVE_SHIFT moves all of these at once.
Silence, or anything that isn't a confident whistle, stops the robot.

Pitch detection and the background-noise gating (SNR vs. a calibrated room
noise floor + a pitch-stability check) come from whistling_world_cup/tone_test.py.

Run: python3 tone_driving.py   (Ctrl+C to quit)
"""

import os
import sys
import time

import numpy as np
import pyaudio

import legoeducation as le

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "whistling_world_cup"))
from tone_test import (  # noqa: E402
    CHUNK, FFT_SIZE, MAX_FREQ, MIN_FREQ, RATE,
    StabilityGate, calibrate_noise, detect_pitch, freq_to_note,
)

# ---- Configuration ----
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "6235"

# Shifts every whistle range together: +1 = one octave higher, -1 = one lower.
# 0 gives F#5 forward / B5 reverse; 1 gives F#6 (1480 Hz) / B6 (1976 Hz).
OCTAVE_SHIFT = 1

FORWARD_NOTE_MIDI = 78 + 12 * OCTAVE_SHIFT  # F#: whistle this to drive straight
REVERSE_NOTE_MIDI = 83 + 12 * OCTAVE_SHIFT  # B and above: drive backward
STEER_RANGE_SEMITONES = 5 # notes within +/- this of the forward note steer
                          # (C# .. A# at 5); notes outside it (and below reverse) stop

DRIVE_SPEED = 80          # forward speed (%) while whistling
REVERSE_SPEED = 45        # backward speed (%) while whistling the reverse note+

# Turning: one fixed speed for every left/right note. Raise it to turn faster.
TURN_SPEED = 5         # added to one wheel and subtracted from the other (%)
FORWARD_RANGE_CENTS = 150  # within this of the forward note drives straight
                           # (150 = +/-1.5 semitones: F .. G around F#)

SMOOTHING = 0.4        # 0 = frozen, 1 = instant; eases speed changes
HOLD_TIME_S = 0.3      # keep the last command this long after the whistle drops,
                       # so taking a breath doesn't make the robot stutter


def note_name(midi):
    """Note name (e.g. 'F#6') for a MIDI number."""
    return freq_to_note(440.0 * 2 ** ((midi - 69) / 12))[0]


def pitch_to_speeds(freq):
    """Map a whistled frequency to (left, right) tank-drive speeds and a label."""
    midi = 69 + 12 * np.log2(freq / 440.0)
    offset = midi - FORWARD_NOTE_MIDI  # semitones sharp (+) or flat (-) of forward note

    if midi >= REVERSE_NOTE_MIDI - 0.5:
        return -REVERSE_SPEED, -REVERSE_SPEED, "REVERSE"
    if abs(offset) > STEER_RANGE_SEMITONES + 0.5:
        return 0.0, 0.0, "out of range"

    if abs(offset) * 100 <= FORWARD_RANGE_CENTS:
        turn = 0.0
    else:
        turn = TURN_SPEED if offset > 0 else -TURN_SPEED

    # Positive turn = right: speed up the left wheel, slow the right.
    left = max(-100.0, min(100.0, DRIVE_SPEED + turn))
    right = max(-100.0, min(100.0, DRIVE_SPEED - turn))
    label = "FORWARD" if turn == 0 else ("RIGHT" if turn > 0 else "LEFT")
    return left, right, label


def main():
    window = np.hanning(CHUNK)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / RATE)
    band = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)

    doublemotor = le.DoubleMotor()
    pa = pyaudio.PyAudio()
    stream = None

    # try/finally so the motor is always stopped, however the script ends.
    try:
        print("Connecting to Double Motor...")
        doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
        if not doublemotor.connected:
            print("Error connecting to Double Motor.")
            return

        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                         input=True, frames_per_buffer=CHUNK)

        print("Calibrating room noise: stay quiet...")
        noise_profile = calibrate_noise(stream, window, band)
        gate = StabilityGate()

        fwd = note_name(FORWARD_NOTE_MIDI)
        print(f"Whistle {fwd} to go forward, higher = right, lower = left, "
              f"{note_name(REVERSE_NOTE_MIDI)} = reverse.")
        print("Ctrl+C to quit.\n")

        smoothed_left = smoothed_right = 0.0
        target_left = target_right = 0.0
        label = "STOPPED"
        last_heard = 0.0

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)

            raw_freq, snr = detect_pitch(samples, window, freqs, band, noise_profile)
            freq = gate.update(raw_freq)
            now = time.time()

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
                # Stop immediately instead of easing down.
                smoothed_left = smoothed_right = 0.0
            else:
                smoothed_left += (target_left - smoothed_left) * SMOOTHING
                smoothed_right += (target_right - smoothed_right) * SMOOTHING

            doublemotor.movement_move_tank(
                speed_left=int(smoothed_left),
                speed_right=int(smoothed_right),
                blocking=False,
            )

            line = f"{heard:<38} {label:<13} L: {smoothed_left:4.0f}%  R: {smoothed_right:4.0f}%"
            print(f"\r{line:<85}", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("Cleaning up...")
        if doublemotor.connected:
            try:
                doublemotor.movement_stop()
            except Exception as exc:
                print(f"Error stopping motor: {exc}")
            doublemotor.disconnect()
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
