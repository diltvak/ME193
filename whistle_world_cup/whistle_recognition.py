"""Recognize the whistled tune in SpongeBob.mp3 when it comes in on the microphone.

At startup the reference recording is decoded and turned into a pitch contour
(one pitch per ~46 ms frame, in semitones). While listening, the microphone is
tracked the same way, and the last couple of seconds of pitch are compared to
the reference with dynamic time warping (DTW), so the tune is still recognized
if it is a bit faster/slower or (optionally) whistled in a different key.

Requires: pip install pyaudio numpy   (macOS: brew install portaudio first)
Decoding the reference uses macOS's built-in `afconvert`.

Run:  python3 whistle_recognition.py              listen on the microphone
      python3 whistle_recognition.py --test F     run the detector over an audio file
"""

import argparse
import shutil
import subprocess
import tempfile
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).with_name("SpongeBob.mp3")

RATE = 44100
CHUNK = 2048          # samples per frame (~46 ms)
FFT_SIZE = CHUNK * 4  # zero-padding for finer peak location

# Whistling is almost a pure sine wave, typically between ~500 Hz and ~4 kHz.
MIN_FREQ = 500
MAX_FREQ = 4500

# A frame counts as "whistle" when it is loud enough and its energy is
# concentrated in one narrow peak.
MIN_RMS = 0.002            # absolute floor (0-1 full scale)
NOISE_MULTIPLIER = 4.0     # ...and this many times the room's calibrated RMS
RELATIVE_GATE = 0.1        # ...and this fraction of the loudest frame in the
RELATIVE_GATE_S = 2.0      # last this many seconds, so echoes/tails of notes
                           # are cut the same way whatever the volume
PURITY_THRESHOLD = 0.5     # fraction of band energy in the peak
CALIBRATION_SECONDS = 1.0  # room noise recorded at startup (stay quiet!)

# Matching. Silent frames are kept in both contours, so rhythm counts too.
TRANSPOSE_INVARIANT = True       # True = whistling it in any key counts
TEMPO_STRETCHES = np.linspace(0.75, 1.33, 9)  # window lengths tried, relative to the reference
MAX_STEP_COST = 3.0              # semitones; caps the penalty for a wild frame
GAP_COST = 1.5                   # penalty for whistle vs. silence in the same step
WARP_COST = 0.3                  # extra penalty for stretching one side (non-diagonal step)
WARP_BAND = 0.15                 # DTW path must stay within this fraction of the diagonal
MATCH_THRESHOLD = 0.5            # avg DTW cost (semitones) at or below = match
MIN_VOICED_RATIO = 0.6           # window must have at least this share of the
                                 # reference's whistled frames
COOLDOWN_S = 2.0                 # ignore new matches this long after one fires


# ---------------------------------------------------------------- audio input

def load_audio(path):
    """Decode any file afconvert understands to mono float32 at RATE."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        src = path
        # SpongeBob.mp3 is really an AAC/M4A file; afconvert trusts the
        # extension, so give it the right one.
        with open(path, "rb") as f:
            if f.read(12)[4:8] == b"ftyp":
                src = Path(tmp) / "in.m4a"
                shutil.copy(path, src)
        out = Path(tmp) / "out.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1",
                        str(src), str(out)], check=True, capture_output=True)
        with wave.open(str(out)) as w:
            data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768


# ------------------------------------------------------------ pitch tracking

_window = np.hanning(CHUNK)
_freqs = np.fft.rfftfreq(FFT_SIZE, 1 / RATE)
_band = (_freqs >= MIN_FREQ) & (_freqs <= MAX_FREQ)


class PitchTracker:
    """Turns audio frames into whistle pitches (MIDI number, or None)."""

    def __init__(self, min_rms=MIN_RMS):
        self.min_rms = min_rms
        self.recent_rms = deque(maxlen=int(RELATIVE_GATE_S * RATE / CHUNK))

    def process(self, samples):
        rms = np.sqrt(np.mean(samples ** 2))
        self.recent_rms.append(rms)
        if rms < max(self.min_rms, RELATIVE_GATE * max(self.recent_rms)):
            return None

        spectrum = np.abs(np.fft.rfft(samples * _window, n=FFT_SIZE))[_band]
        power = spectrum ** 2
        peak = int(np.argmax(power))
        if power[max(peak - 6, 0):peak + 7].sum() / power.sum() < PURITY_THRESHOLD:
            return None

        # Parabolic interpolation around the peak for sub-bin accuracy.
        offset = 0.0
        if 0 < peak < len(spectrum) - 1:
            a, b, c = np.log(spectrum[peak - 1:peak + 2] + 1e-12)
            offset = 0.5 * (a - c) / (a - 2 * b + c)
        freq = _freqs[_band][peak] + offset * (_freqs[1] - _freqs[0])
        return 69 + 12 * np.log2(freq / 440.0)


def pitch_track(audio):
    """Per-frame pitch (MIDI or None) of a whole recording."""
    tracker = PitchTracker()
    return [tracker.process(audio[i:i + CHUNK])
            for i in range(0, len(audio) - CHUNK + 1, CHUNK)]


def build_template(path):
    """Reference contour from first to last whistled frame (NaN = silence)."""
    track = to_array(pitch_track(load_audio(path)))
    voiced = np.flatnonzero(~np.isnan(track))
    if len(voiced) == 0:
        raise SystemExit(f"No whistle found in {path}")
    return track[voiced[0]:voiced[-1] + 1]


# ------------------------------------------------------------------ matching

def to_array(pitches):
    return np.array([np.nan if p is None else p for p in pitches])


def normalize(seq):
    """Shift a contour so its median whistled pitch is 0 (if key doesn't matter)."""
    return seq - np.nanmedian(seq) if TRANSPOSE_INVARIANT else seq


def dtw_cost(a, b):
    """Average per-step DTW distance between two contours (NaN = silence)."""
    n, m = len(a), len(b)
    local = np.minimum(np.abs(a[:, None] - b[None, :]), MAX_STEP_COST)
    silent = np.isnan(a)[:, None] | np.isnan(b)[None, :]
    both_silent = np.isnan(a)[:, None] & np.isnan(b)[None, :]
    local[silent] = GAP_COST
    local[both_silent] = 0.0

    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1))
    band = WARP_BAND * max(n, m)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(i / n - j / m) * max(n, m) > band:
                continue
            best, count = min((acc[i - 1, j - 1], steps[i - 1, j - 1]),
                              (acc[i - 1, j] + WARP_COST, steps[i - 1, j]),
                              (acc[i, j - 1] + WARP_COST, steps[i, j - 1]))
            acc[i, j] = best + local[i - 1, j - 1]
            steps[i, j] = count + 1
    return acc[n, m] / steps[n, m]


class TuneMatcher:
    """Feed one frame's pitch at a time; reports when the tune has just ended."""

    def __init__(self, template):
        self.template = normalize(template)
        self.template_voiced = np.count_nonzero(~np.isnan(template))
        self.history = deque(maxlen=int(len(template) * max(TEMPO_STRETCHES)) + 2)
        self.cooldown_until = -np.inf

    def update(self, pitch, now):
        """Return the match cost if the tune was just recognized, else None."""
        self.history.append(pitch)
        # Only check right as a whistle ends, i.e. the tune's last note finished.
        if pitch is not None or len(self.history) < 2 or self.history[-2] is None:
            return None
        if now < self.cooldown_until:
            return None

        best = self.best_cost()
        if best is not None and best <= MATCH_THRESHOLD:
            self.cooldown_until = now + COOLDOWN_S
            self.history.clear()
            return best
        return None

    def best_cost(self):
        """Lowest DTW cost over the tempo stretches, or None if not enough whistle."""
        frames = to_array(list(self.history)[:-1])  # drop the trailing silent frame
        best = None
        for stretch in TEMPO_STRETCHES:
            window = frames[-int(round(len(self.template) * stretch)):]
            voiced = window[~np.isnan(window)]
            if len(voiced) < MIN_VOICED_RATIO * self.template_voiced * stretch:
                continue
            # Start the window at its first whistled frame, like the template.
            window = window[np.flatnonzero(~np.isnan(window))[0]:]
            cost = dtw_cost(normalize(window), self.template)
            best = cost if best is None else min(best, cost)
        return best


# --------------------------------------------------------------------- modes

def announce(cost, when):
    print(f"\n*** SpongeBob whistle recognized at {when}  (cost {cost:.2f}) ***")


def run_file(path, matcher):
    """Offline check: stream a recording through the matcher frame by frame."""
    audio = load_audio(path)
    hits = 0
    for k, pitch in enumerate(pitch_track(audio)):
        t = (k + 1) * CHUNK / RATE
        cost = matcher.update(pitch, t)
        if cost is not None:
            hits += 1
            announce(cost, f"{t:.2f} s")
    print(f"{path}: {hits} match(es)")


def run_microphone(matcher):
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)

    def read():
        return np.frombuffer(stream.read(CHUNK, exception_on_overflow=False),
                             dtype=np.float32)

    print(f"Calibrating: stay quiet for {CALIBRATION_SECONDS:g} s...")
    n = max(1, int(CALIBRATION_SECONDS * RATE / CHUNK))
    noise_rms = np.sqrt(np.mean(np.concatenate([read() for _ in range(n)]) ** 2))
    tracker = PitchTracker(max(MIN_RMS, NOISE_MULTIPLIER * noise_rms))

    print("Listening for the SpongeBob whistle (Ctrl+C to stop)\n")
    try:
        while True:
            pitch = tracker.process(read())
            cost = matcher.update(pitch, time.monotonic())
            status = f"whistle {pitch:5.1f} (MIDI)" if pitch is not None else "..."
            print(f"\r{status:<30}", end="", flush=True)
            if cost is not None:
                announce(cost, time.strftime("%H:%M:%S"))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def main():
    parser = argparse.ArgumentParser(description="Recognize the SpongeBob whistle on the microphone.")
    parser.add_argument("--test", nargs="+", metavar="FILE",
                        help="run the detector over audio files instead of the mic")
    args = parser.parse_args()

    template = build_template(REFERENCE)
    print(f"Reference: {np.count_nonzero(~np.isnan(template))} whistled frames "
          f"over {len(template) * CHUNK / RATE:.2f} s")

    if args.test:
        for path in args.test:
            run_file(path, TuneMatcher(template))
    else:
        run_microphone(TuneMatcher(template))


if __name__ == "__main__":
    main()
